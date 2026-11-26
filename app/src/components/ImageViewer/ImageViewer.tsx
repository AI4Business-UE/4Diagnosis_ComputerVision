import { useRef, useEffect, useState, useCallback } from 'react';
import './ImageViewer.css';
import type { Glomeruli, SlideInfo, SlicesSummary, TileInfo } from '../../services/api';
import { formatApiError } from '../../services/errors';
import type { ImageVersion, ImageVersionId } from '../../types/Sample';

declare global {
    interface Window {
        Tiff: any;
    }
}

let tiffInitialized = false;
function ensureTiffInitialized() {
    if (!tiffInitialized && window.Tiff?.initialize) {
        try {
            window.Tiff.initialize({ TOTAL_MEMORY: 512 * 1024 * 1024 });
        } catch {}
        tiffInitialized = true;
    }
}

interface ViewState {
    scale: number;
    panX: number;
    panY: number;
}

/** Matches backend/cv/cv/settings.py FIBROSIS_THRESHOLD — fallback until a real result arrives. */
const DEFAULT_FIBROSIS_THRESHOLD = 0.4;
/** Outside this band the B-channel threshold has no clinically meaningful effect. */
const FIBROSIS_THRESHOLD_MIN = 0.3;
const FIBROSIS_THRESHOLD_MAX = 0.5;
const clampFibrosisThreshold = (v: number) => Math.min(FIBROSIS_THRESHOLD_MAX, Math.max(FIBROSIS_THRESHOLD_MIN, v));

interface ImageViewerProps {
    versions: ImageVersion[];
    activeVersionId?: ImageVersionId | null;
    onActiveVersionChange?: (id: ImageVersionId | null) => void;
    hasActiveSample?: boolean;
    isConverting?: boolean;
    slicesSummary?: SlicesSummary | null;
    groupingUpdating?: boolean;
    onToggleGrouping?: (grouped: boolean) => void;
    glomeruli?: Glomeruli[];
    slideInfo?: SlideInfo | null;
    tilesScanned?: TileInfo[];
    confThresholds?: { 0: number; 1: number };
    onConfThresholdsChange?: (v: { 0: number; 1: number }) => void;
    /** Current fibrosis ratio (0-1), drawn as a bar over the overlay. */
    fibrosisRatio?: number;
    /** B-channel threshold (0-1) last used to compute fibrosisRatio. */
    fibrosisThreshold?: number;
    /** True while a threshold-driven re-analysis is in flight. */
    fibrosisRecalculating?: boolean;
    /** Fired once the user releases the threshold handle — triggers a backend re-analysis. */
    onFibrosisThresholdCommit?: (threshold: number) => void;
}

const SHORT_VERSION_LABELS: Record<ImageVersionId, string> = {
    raw: 'Raw',
    cleaned: 'Wyczyszczony',
    merged: 'Zgrupowany',
    fibrosis: 'Zwłóknienie',
    length: 'Długość',
    glomeruli: 'Kłębuszki',
};

export default function ImageViewer({
        versions, activeVersionId, onActiveVersionChange,
        hasActiveSample = false,
        isConverting = false,
        slicesSummary, groupingUpdating = false, onToggleGrouping,
        glomeruli, slideInfo, tilesScanned,
        confThresholds = { 0: 0.15, 1: 0.15 }, onConfThresholdsChange,
        fibrosisThreshold, fibrosisRecalculating = false, onFibrosisThresholdCommit,
    }: ImageViewerProps) {
    const [advancedOpen, setAdvancedOpen] = useState(false);
    const [showTiles, setShowTiles] = useState(true);
    const committedFibrosisThreshold = clampFibrosisThreshold(fibrosisThreshold ?? DEFAULT_FIBROSIS_THRESHOLD);
    const [fibrosisThresholdDraft, setFibrosisThresholdDraft] = useState(committedFibrosisThreshold);
    useEffect(() => { setFibrosisThresholdDraft(committedFibrosisThreshold); }, [committedFibrosisThreshold]);
    const commitFibrosisThreshold = () => {
        if (fibrosisThresholdDraft !== committedFibrosisThreshold) onFibrosisThresholdCommit?.(fibrosisThresholdDraft);
    };
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [loadError, setLoadError] = useState<string | null>(null);
    const [reloadTick, setReloadTick] = useState(0);
    const [isImageLoaded, setIsImageLoaded] = useState(false);
    const [activeIndex, setActiveIndex] = useState(0);

    const sourceImageRef = useRef<HTMLCanvasElement | HTMLImageElement | null>(null);
    const originalDimsRef = useRef<{ width: number; height: number } | null>(null);
    const prevVersionIdsRef = useRef<string[]>([]);

    const releaseSource = useCallback(() => {
        const prev = sourceImageRef.current;
        if (prev && prev instanceof HTMLCanvasElement) {
            prev.width = 0;
            prev.height = 0;
        }
        sourceImageRef.current = null;
    }, []);

    const [view, setView] = useState<ViewState>({
        scale: 1,
        panX: 0,
        panY: 0,
    });

    const isDragging = useRef(false);
    const [dragging, setDragging] = useState(false);
    const lastMousePos = useRef({ x: 0, y: 0 });

    const safeIndex = versions.length > 0 ? Math.min(activeIndex, versions.length - 1) : 0;
    const activeVersion = versions[safeIndex] ?? null;
    const imageUrl = activeVersion?.url ?? null;

    const isBitmapUrl = (url: string | null) => /\.(jpg|jpeg|png)(\/?|\?|$)/i.test(url ?? '');

    const selectIndex = useCallback((idx: number) => {
        setActiveIndex(idx);
        onActiveVersionChange?.(versions[idx]?.id ?? null);
    }, [versions, onActiveVersionChange]);

    const goPrev = useCallback(() => {
        if (versions.length <= 1) return;
        const nextIdx = (safeIndex - 1 + versions.length) % versions.length;
        selectIndex(nextIdx);
    }, [versions.length, safeIndex, selectIndex]);

    const goNext = useCallback(() => {
        if (versions.length <= 1) return;
        const nextIdx = (safeIndex + 1) % versions.length;
        selectIndex(nextIdx);
    }, [versions.length, safeIndex, selectIndex]);

    useEffect(() => {
        if (!activeVersionId) return;
        const idx = versions.findIndex(v => v.id === activeVersionId);
        if (idx !== -1) {
            setActiveIndex(prev => (prev !== idx ? idx : prev));
        }
    }, [activeVersionId, versions]);

    useEffect(() => {
        const prevIds = prevVersionIdsRef.current;
        const curIds = versions.map(v => v.id);
        prevVersionIdsRef.current = curIds;

        if (versions.length === 0) {
            setActiveIndex(0);
            onActiveVersionChange?.(null);
            return;
        }

        // Auto-navigate to glomeruli when it appears for the first time
        const gIdx = versions.findIndex(v => v.id === 'glomeruli');
        if (gIdx !== -1 && !prevIds.includes('glomeruli')) {
            setActiveIndex(gIdx);
            onActiveVersionChange?.('glomeruli');
            return;
        }

        const mergedChanged = prevIds.includes('merged') !== curIds.includes('merged');
        const initialLoad = prevIds.length === 0 && curIds.length > 0;
        if (initialLoad || mergedChanged) {
            const mergedIdx = versions.findIndex(v => v.id === 'merged');
            if (mergedIdx !== -1) {
                setActiveIndex(mergedIdx);
                onActiveVersionChange?.('merged');
                return;
            }
            const cleanedIdx = versions.findIndex(v => v.id === 'cleaned');
            if (cleanedIdx !== -1) {
                setActiveIndex(cleanedIdx);
                onActiveVersionChange?.('cleaned');
                return;
            }
            setActiveIndex(0);
            onActiveVersionChange?.(versions[0]?.id ?? null);
            return;
        }

        if (activeIndex > versions.length - 1) {
            const preferredIdx =
                versions.findIndex(v => v.id === 'merged') !== -1
                    ? versions.findIndex(v => v.id === 'merged')
                    : versions.findIndex(v => v.id === 'cleaned') !== -1
                        ? versions.findIndex(v => v.id === 'cleaned')
                        : 0;
            setActiveIndex(preferredIdx);
            onActiveVersionChange?.(versions[preferredIdx]?.id ?? null);
        }
    }, [versions]);

    const drawTiff = useCallback((currentView: ViewState) => {
        const canvas = canvasRef.current;
        const source = sourceImageRef.current;
        const dims = originalDimsRef.current;

        if (!canvas || !source || !dims) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const containerWidth = Math.max(1, containerRef.current?.clientWidth ?? 800);
        const containerHeight = Math.max(1, containerRef.current?.clientHeight ?? 600);

        if (canvas.width !== containerWidth || canvas.height !== containerHeight) {
            canvas.width = containerWidth;
            canvas.height = containerHeight;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.save();
        ctx.translate(currentView.panX, currentView.panY);
        ctx.scale(currentView.scale, currentView.scale);
        ctx.drawImage(source, 0, 0);
        ctx.restore();
    }, []);

    useEffect(() => {
        let isMounted = true;
        const controller = new AbortController();
        let objectUrlToRevoke: string | null = null;

        const load = async () => {
            if (!isMounted) return;

            setIsLoading(true);
            setIsImageLoaded(false);
            setLoadError(null);
            releaseSource();
            originalDimsRef.current = null;

            if (!imageUrl) {
                if (isMounted) {
                    setIsLoading(false);
                }
                return;
            }

            try {
                const fetchUrl = imageUrl.includes('?')
                    ? `${imageUrl}&preview=1`
                    : `${imageUrl}?preview=1`;

                const response = await fetch(fetchUrl, { signal: controller.signal });
                if (!response.ok) {
                    let detail = `HTTP ${response.status}`;
                    try {
                        const payload = await response.json();
                        if (typeof payload?.error === 'string' && payload.error.trim()) {
                            detail = payload.error;
                        }
                    } catch {}
                    throw new Error(detail);
                }

                const contentType = (response.headers.get('content-type') || '').toLowerCase();
                const isStandardBitmap =
                    contentType.includes('image/jpeg') ||
                    contentType.includes('image/png') ||
                    isBitmapUrl(imageUrl);

                if (isStandardBitmap) {
                    const blob = await response.blob();
                    if (!isMounted) return;
                    const objectUrl = URL.createObjectURL(blob);
                    objectUrlToRevoke = objectUrl;

                    const img = new Image();
                    img.onload = () => {
                        if (!isMounted) {
                            URL.revokeObjectURL(objectUrl);
                            return;
                        }

                        releaseSource();
                        sourceImageRef.current = img;
                        originalDimsRef.current = { width: img.naturalWidth, height: img.naturalHeight };

                        URL.revokeObjectURL(objectUrl);
                        if (objectUrlToRevoke === objectUrl) objectUrlToRevoke = null;
                        setIsImageLoaded(true);
                        setIsLoading(false);
                    };

                    img.onerror = () => {
                        URL.revokeObjectURL(objectUrl);
                        if (objectUrlToRevoke === objectUrl) objectUrlToRevoke = null;
                        if (isMounted) {
                            setLoadError('Przeglądarka nie mogła zdekodować obrazu (uszkodzony lub zbyt duży plik).');
                            setIsLoading(false);
                        }
                    };

                    img.src = objectUrl;
                    return;
                }

                if (!window.Tiff) {
                    if (isMounted) {
                        setLoadError('Brak dekodera TIFF (Tiff.js nie został załadowany).');
                        setIsLoading(false);
                    }
                    return;
                }

                const buffer = await response.arrayBuffer();
                if (!isMounted) return;

                ensureTiffInitialized();
                const tiff = new window.Tiff({ buffer });
                let generatedCanvas: HTMLCanvasElement | null = null;
                try {
                    tiff.setDirectory(0);
                    generatedCanvas = tiff.toCanvas();
                } finally {
                    try { tiff.close(); } catch {}
                }

                if (!isMounted) {
                    if (generatedCanvas) {
                        generatedCanvas.width = 0;
                        generatedCanvas.height = 0;
                    }
                    return;
                }

                if (generatedCanvas) {
                    releaseSource();
                    sourceImageRef.current = generatedCanvas;
                    originalDimsRef.current = {
                        width: generatedCanvas.width,
                        height: generatedCanvas.height,
                    };

                    setIsImageLoaded(true);
                    setLoadError(null);
                } else {
                    setLoadError('Nie udało się zdekodować pliku TIFF (obraz może przekraczać limit pamięci przeglądarki).');
                }
            } catch (e: unknown) {
                if (controller.signal.aborted || !isMounted) return;
                setLoadError(formatApiError(e, 'Nie udało się załadować obrazu.'));
            } finally {
                if (isMounted) setIsLoading(false);
            }
        };

        load();

        return () => {
            isMounted = false;
            controller.abort();
            if (objectUrlToRevoke) {
                URL.revokeObjectURL(objectUrlToRevoke);
            }
        };
    }, [imageUrl, reloadTick, releaseSource]);

    useEffect(() => () => releaseSource(), [releaseSource]);

    useEffect(() => {
        if (!isImageLoaded || !originalDimsRef.current || !containerRef.current) return;

        const dims = originalDimsRef.current;
        const containerWidth = Math.max(1, containerRef.current.clientWidth);
        const containerHeight = Math.max(1, containerRef.current.clientHeight);

        const fitScale = Math.min(containerWidth / dims.width, containerHeight / dims.height);
        const centeredPanX = (containerWidth - dims.width * fitScale) / 2;
        const centeredPanY = (containerHeight - dims.height * fitScale) / 2;

        setView({
            scale: fitScale,
            panX: centeredPanX,
            panY: centeredPanY,
        });
    }, [isImageLoaded, imageUrl]);

    useEffect(() => {
        if (isImageLoaded) {
            drawTiff(view);
        }
    }, [view, isImageLoaded, drawTiff]);

    const handleWheel = useCallback((event: WheelEvent) => {
        event.preventDefault();
        const canvas = canvasRef.current;
        if (!canvas || !isImageLoaded) return;

        const zoomFactor = -event.deltaY * 0.001;
        const maxScale = 6.0;
        const minScale = 0.05;

        const newScale = Math.max(minScale, Math.min(view.scale + zoomFactor, maxScale));
        if (newScale === view.scale) return;

        const rect = canvas.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;

        const oldImageX = (mouseX - view.panX) / view.scale;
        const oldImageY = (mouseY - view.panY) / view.scale;

        const newPanX = mouseX - oldImageX * newScale;
        const newPanY = mouseY - oldImageY * newScale;

        setView({
            scale: newScale,
            panX: newPanX,
            panY: newPanY,
        });
    }, [view, isImageLoaded]);

    const handleMouseDown = useCallback((event: MouseEvent) => {
        if (event.button !== 0 || !isImageLoaded) return;
        isDragging.current = true;
        setDragging(true);
        lastMousePos.current = { x: event.clientX, y: event.clientY };
    }, [isImageLoaded]);

    const handleMouseMove = useCallback((event: MouseEvent) => {
        if (!isDragging.current) return;

        const dx = event.clientX - lastMousePos.current.x;
        const dy = event.clientY - lastMousePos.current.y;

        setView(prevView => ({
            ...prevView,
            panX: prevView.panX + dx,
            panY: prevView.panY + dy,
        }));

        lastMousePos.current = { x: event.clientX, y: event.clientY };
    }, []);

    const handleMouseUp = useCallback(() => {
        if (isDragging.current) {
            isDragging.current = false;
            setDragging(false);
        }
    }, []);

    useEffect(() => {
        const canvas = canvasRef.current;

        if (canvas) {
            canvas.addEventListener('wheel', handleWheel, { passive: false });
            canvas.addEventListener('mousedown', handleMouseDown);

            window.addEventListener('mousemove', handleMouseMove);
            window.addEventListener('mouseup', handleMouseUp);
        }

        return () => {
            if (canvas) {
                canvas.removeEventListener('wheel', handleWheel);
                canvas.removeEventListener('mousedown', handleMouseDown);
            }
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        };
    }, [handleWheel, handleMouseDown, handleMouseMove, handleMouseUp]);

    const displayScale = view.scale;

    const renderGlomeruliOverlay = () => {
        if (activeVersion?.id !== 'glomeruli') return null;
        if (!slideInfo || !originalDimsRef.current) return null;

        const tiffW = originalDimsRef.current.width;
        const tiffH = originalDimsRef.current.height;
        const scaleX = tiffW / slideInfo.w;
        const scaleY = tiffH / slideInfo.h;

        return (
            <svg
                style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    pointerEvents: 'none',
                    overflow: 'hidden',
                }}
            >
                {/* Kafelki przeskanowane — teal=tkanka, szary=tło/szkło */}
                {showTiles && tilesScanned?.map((t, i) => {
                    const x = t.x * scaleX * view.scale + view.panX;
                    const y = t.y * scaleY * view.scale + view.panY;
                    const w = t.w * scaleX * view.scale;
                    const h = t.h * scaleY * view.scale;
                    return (
                        <rect
                            key={`tile-${i}`}
                            x={x}
                            y={y}
                            width={w}
                            height={h}
                            fill={t.tissue ? 'rgba(0,188,212,0.18)' : 'rgba(150,150,150,0.10)'}
                            stroke={t.tissue ? 'rgba(0,188,212,0.5)' : 'rgba(150,150,150,0.25)'}
                            strokeWidth={0.8}
                        />
                    );
                })}

                {/* Bounding boxy kłębuszków — filtrowane przez próg ufności per klasa */}
                {glomeruli?.filter(g => g.conf >= confThresholds[g.cls as 0 | 1]).map((g, i) => {
                    const x = g.x1 * scaleX * view.scale + view.panX;
                    const y = g.y1 * scaleY * view.scale + view.panY;
                    const w = (g.x2 - g.x1) * scaleX * view.scale;
                    const h = (g.y2 - g.y1) * scaleY * view.scale;
                    const color = g.cls === 0 ? '#00e676' : '#ff1744';
                    const label = `${g.cls === 0 ? 'niezwłókniony' : 'zwłókniony'} ${(g.conf * 100).toFixed(0)}%`;
                    const showLabel = w > 30;
                    return (
                        <g key={`g-${i}`}>
                            <rect x={x} y={y} width={w} height={h} fill="none" stroke={color} strokeWidth={1.5} />
                            {showLabel && (
                                <>
                                    <rect
                                        x={x}
                                        y={y - 16}
                                        width={label.length * 6.5}
                                        height={15}
                                        fill="rgba(0,0,0,0.6)"
                                        rx={2}
                                    />
                                    <text
                                        x={x + 3}
                                        y={y - 4}
                                        fill={color}
                                        fontSize={11}
                                        fontFamily="monospace"
                                    >
                                        {label}
                                    </text>
                                </>
                            )}
                        </g>
                    );
                })}
            </svg>
        );
    };

    const emptyHint = !imageUrl && !isConverting
        ? (hasActiveSample ? 'Kliknij „Konwertuj”, aby rozpocząć.' : 'Wybierz folder, aby rozpocząć.')
        : null;

    return (
        <div className="image-viewer-wrapper">
            <div className={`viewer-header-bar ${versions.length > 0 ? '' : 'hidden'}`}>
                <div className="version-tabs" role="tablist" aria-label="Widoki preparatu">
                    {versions.map((v, idx) => (
                        <button
                            key={v.id}
                            type="button"
                            role="tab"
                            aria-selected={idx === safeIndex}
                            className={`version-tab${idx === safeIndex ? ' active' : ''}`}
                            onClick={() => selectIndex(idx)}
                            title={v.label}
                        >
                            {SHORT_VERSION_LABELS[v.id] ?? v.label}
                        </button>
                    ))}
                </div>

                {slicesSummary && (
                    <div className="grouping-status-bar">
                        {slicesSummary.is_grouped && (
                            <>
                                <span className="grouping-badge grouped">
                                    Wykryto grupy: {slicesSummary.n_slices}
                                </span>
                                {onToggleGrouping && (
                                    <button
                                        type="button"
                                        className="ungroup-button"
                                        disabled={groupingUpdating}
                                        onClick={() => onToggleGrouping(false)}
                                    >
                                        {groupingUpdating ? 'Przełączanie...' : 'Nie grupuj'}
                                    </button>
                                )}
                            </>
                        )}

                        {slicesSummary.user_ungrouped && (
                            <>
                                <span className="grouping-badge ungrouped">
                                    Grupowanie wyłączone
                                </span>
                                {onToggleGrouping && (
                                    <button
                                        type="button"
                                        className="ungroup-button restore"
                                        disabled={groupingUpdating}
                                        onClick={() => onToggleGrouping(true)}
                                    >
                                        {groupingUpdating ? 'Przełączanie...' : 'Przywróć grupowanie'}
                                    </button>
                                )}
                            </>
                        )}

                        {!slicesSummary.is_grouped && !slicesSummary.user_ungrouped && slicesSummary.fallback_used && (
                            <span className="grouping-badge fallback">
                                Nie wykryto grup
                            </span>
                        )}
                    </div>
                )}
            </div>

            <div className="viewer-stage">
                {versions.length > 1 && (
                    <button className="viewer-nav viewer-nav-left" onClick={goPrev} aria-label="Poprzednia wersja">
                        {'<'}
                    </button>
                )}

                <div className="canvas-container" ref={containerRef}>
                    {isConverting && (
                        <div className="viewer-converting-overlay" role="status" aria-live="polite">
                            <div className="viewer-spinner" />
                            <div className="viewer-converting-title">Konwersja preparatu...</div>
                        </div>
                    )}

                    {isLoading && Boolean(imageUrl) && !isImageLoaded && !loadError && !isConverting && (
                        <div className="viewer-converting-overlay" role="status" aria-live="polite">
                            <div className="viewer-spinner" />
                            <div className="viewer-converting-title">Ładowanie obrazu...</div>
                        </div>
                    )}

                    {emptyHint && !loadError && (
                        <div className="viewer-empty-state">
                            <span>{emptyHint}</span>
                        </div>
                    )}

                    {loadError && !isConverting && (
                        <div className="viewer-error-overlay" role="alert">
                            <div className="viewer-error-card">
                                <div className="viewer-error-badge">BŁĄD</div>
                                <div className="viewer-error-title">Nie udało się załadować obrazu</div>
                                <div className="viewer-error-detail">{loadError}</div>
                                <button
                                    type="button"
                                    className="viewer-retry-btn"
                                    onClick={() => setReloadTick(t => t + 1)}
                                >
                                    Spróbuj ponownie
                                </button>
                            </div>
                        </div>
                    )}

                    <canvas
                        ref={canvasRef}
                        className="tiff-canvas"
                        style={{
                            display: (loadError || !isImageLoaded || isConverting || !imageUrl) ? 'none' : 'block',
                            cursor: isImageLoaded ? (dragging ? 'move' : 'default') : 'default',
                        }}
                    />

                    {isImageLoaded && !loadError && renderGlomeruliOverlay()}

                    {/* Zoom on the left, glomeruli controls on the right, sharing one baseline. */}
                    <div className={`viewer-toolbar ${isImageLoaded && !loadError && !isConverting ? '' : 'hidden'}`}>
                        <div className="scale-badge" title="Zbliżenie">
                            <span className="scale-value">{Math.round(displayScale * 100)}%</span>
                            <span className="scale-caption">zoom</span>
                        </div>

                        {isImageLoaded && activeVersion?.id === 'fibrosis' && (
                            <div className="conf-slider-overlay">
                                <div className="conf-slider-row">
                                    <span className="conf-label">Próg czułości</span>
                                    {fibrosisRecalculating ? (
                                        <span className="fibrosis-recalculating-label">Przeliczanie...</span>
                                    ) : (
                                        <input
                                            className="conf-range range-fibrosis"
                                            type="range"
                                            min={FIBROSIS_THRESHOLD_MIN * 100}
                                            max={FIBROSIS_THRESHOLD_MAX * 100}
                                            step={0.1}
                                            value={(fibrosisThresholdDraft * 100).toFixed(1)}
                                            onChange={e => setFibrosisThresholdDraft(Number(e.target.value) / 100)}
                                            onMouseUp={commitFibrosisThreshold}
                                            onTouchEnd={commitFibrosisThreshold}
                                            onKeyUp={commitFibrosisThreshold}
                                            aria-label="Próg czułości wykrywania zwłóknienia"
                                        />
                                    )}
                                    <span className="conf-value">
                                        {(fibrosisThresholdDraft * 100).toFixed(1)}%
                                    </span>
                                </div>
                            </div>
                        )}

                        {isImageLoaded && activeVersion?.id === 'glomeruli' && (
                            <div className="conf-slider-overlay">
                                {!advancedOpen ? (
                                    /* Tryb prosty */
                                    <div className="conf-slider-row">
                                        <span className="conf-label">Próg ufności</span>
                                        {/* min = conf modelu z backendu (slideInfo.conf).
                                            Poniżej tego progu model nie zwraca detekcji,
                                            więc suwak niżej nic nie zmienia. Fallback 15
                                            tylko gdy slideInfo jeszcze nie dotarło. */}
                                        <input
                                            className="conf-range"
                                            type="range"
                                            min={Math.round((slideInfo?.conf ?? 0.15) * 100)}
                                            max={100}
                                            value={Math.round(confThresholds[0] * 100)}
                                            onChange={e => {
                                                const v = Number(e.target.value) / 100;
                                                onConfThresholdsChange?.({ 0: v, 1: v });
                                            }}
                                        />
                                        <span className="conf-value">
                                            {Math.round(confThresholds[0] * 100)}%
                                        </span>
                                        <span className="conf-divider" />
                                        <button className="conf-advanced-toggle" onClick={() => setAdvancedOpen(true)}>
                                            Zaawansowane ▾
                                        </button>
                                        <button
                                            className={`conf-advanced-toggle ${showTiles ? 'active' : ''}`}
                                            onClick={() => setShowTiles(t => !t)}
                                        >
                                            {showTiles ? 'Ukryj kafelki' : 'Pokaż kafelki'}
                                        </button>
                                    </div>
                                ) : (
                                    /* Tryb zaawansowany */
                                    <div className="conf-advanced">
                                        <div className="conf-slider-row">
                                            <span className="conf-label conf-label-healthy">Niezwłókniony</span>
                                            <input
                                                className="conf-range range-healthy"
                                                type="range"
                                                min={Math.round((slideInfo?.conf ?? 0.15) * 100)}
                                                max={100}
                                                value={Math.round(confThresholds[0] * 100)}
                                                onChange={e => onConfThresholdsChange?.({ ...confThresholds, 0: Number(e.target.value) / 100 })}
                                            />
                                            <span className="conf-value">
                                                {Math.round(confThresholds[0] * 100)}%
                                            </span>
                                        </div>
                                        <div className="conf-slider-row">
                                            <span className="conf-label conf-label-sclerotic">Zwłókniony</span>
                                            <input
                                                className="conf-range range-sclerotic"
                                                type="range"
                                                min={Math.round((slideInfo?.conf ?? 0.15) * 100)}
                                                max={100}
                                                value={Math.round(confThresholds[1] * 100)}
                                                onChange={e => onConfThresholdsChange?.({ ...confThresholds, 1: Number(e.target.value) / 100 })}
                                            />
                                            <span className="conf-value">
                                                {Math.round(confThresholds[1] * 100)}%
                                            </span>
                                        </div>
                                        <div className="conf-slider-row conf-actions">
                                            <button
                                                className="conf-advanced-toggle"
                                                onClick={() => {
                                                    const v = Math.max(confThresholds[0], confThresholds[1]);
                                                    onConfThresholdsChange?.({ 0: v, 1: v });
                                                    setAdvancedOpen(false);
                                                }}
                                            >
                                                ← Jeden próg
                                            </button>
                                            <button
                                                className={`conf-advanced-toggle ${showTiles ? 'active' : ''}`}
                                                onClick={() => setShowTiles(t => !t)}
                                            >
                                                {showTiles ? 'Ukryj kafelki' : 'Pokaż kafelki'}
                                            </button>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>

                {versions.length > 1 && (
                    <button className="viewer-nav viewer-nav-right" onClick={goNext} aria-label="Następna wersja">
                        {'>'}
                    </button>
                )}
            </div>
        </div>
    );
}