import { useRef, useEffect, useState, useCallback } from 'react';
import './ImageViewer.css';
import type { Glomerulus, SlideInfo, TileInfo } from '../../services/api';

declare global {
    interface Window {
        Tiff: any;
    }
}

interface ViewState {
    scale: number;
    panX: number;
    panY: number;
}

interface ImageViewerProps {
    versions: Array<{
        id: 'original' | 'fibrosis' | 'length' | 'glomeruli';
        label: string;
        url: string;
    }>;
    glomeruli?: Glomerulus[];
    slideInfo?: SlideInfo | null;
    tilesScanned?: TileInfo[];
    confThresholds?: { 0: number; 1: number };
    onConfThresholdsChange?: (v: { 0: number; 1: number }) => void;
}

export default function ImageViewer({ versions, glomeruli, slideInfo, tilesScanned, confThresholds = { 0: 0.15, 1: 0.15 }, onConfThresholdsChange }: ImageViewerProps) {
    const [advancedOpen, setAdvancedOpen] = useState(false);
    const [showTiles, setShowTiles] = useState(true);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [isImageLoaded, setIsImageLoaded] = useState(false);
    const [activeIndex, setActiveIndex] = useState(0);

    const sourceCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const originalDimsRef = useRef<{ width: number; height: number } | null>(null);
    const prevVersionIdsRef = useRef<string[]>([]);

    const [view, setView] = useState<ViewState>({
        scale: 1,
        panX: 0,
        panY: 0,
    });

    const isDragging = useRef(false);
    const lastMousePos = useRef({ x: 0, y: 0 });

    const activeVersion = versions[activeIndex] ?? null;
    const imageUrl = activeVersion?.url ?? null;

    const isBitmapUrl = (url: string | null) => /\.(jpg|jpeg|png)\/?$/i.test(url ?? '');

    const goPrev = useCallback(() => {
        if (versions.length <= 1) return;
        setActiveIndex(prev => (prev - 1 + versions.length) % versions.length);
    }, [versions.length]);

    const goNext = useCallback(() => {
        if (versions.length <= 1) return;
        setActiveIndex(prev => (prev + 1) % versions.length);
    }, [versions.length]);

    useEffect(() => {
        const prevIds = prevVersionIdsRef.current;
        const curIds = versions.map(v => v.id);
        prevVersionIdsRef.current = curIds;

        if (versions.length === 0) { setActiveIndex(0); return; }
        if (activeIndex > versions.length - 1) { setActiveIndex(0); return; }

        // Auto-nawiguj do widoku kłębuszków gdy wersja pojawia się po raz pierwszy
        const gIdx = versions.findIndex(v => v.id === 'glomeruli');
        if (gIdx !== -1 && !prevIds.includes('glomeruli')) {
            setActiveIndex(gIdx);
        }
    }, [versions]);

    const drawTiff = useCallback((currentView: ViewState) => {
        const canvas = canvasRef.current;
        const sourceCanvas = sourceCanvasRef.current;
        const dims = originalDimsRef.current;

        if (!canvas || !sourceCanvas || !dims) return;

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
        ctx.drawImage(sourceCanvas, 0, 0);
        ctx.restore();
    }, []);

    useEffect(() => {
        let isMounted = true;

        const load = async () => {
            if (!isMounted) return;

            setIsLoading(true);
            setIsImageLoaded(false);
            setError(null);

            if (!imageUrl) {
                if (isMounted) {
                    setError('Brak wygenerowanego obrazu.');
                    setIsLoading(false);
                }
                return;
            }

            try {
                if (isBitmapUrl(imageUrl)) {
                    const response = await fetch(imageUrl);
                    if (!response.ok) throw new Error(`Nie udało się pobrać obrazu (${response.status})`);

                    const blob = await response.blob();
                    const objectUrl = URL.createObjectURL(blob);

                    const img = new Image();
                    img.onload = () => {
                        if (!isMounted) {
                            URL.revokeObjectURL(objectUrl);
                            return;
                        }

                        const canvas = document.createElement('canvas');
                        canvas.width = img.width;
                        canvas.height = img.height;

                        const ctx = canvas.getContext('2d');
                        if (!ctx) {
                            URL.revokeObjectURL(objectUrl);
                            setError('Brak kontekstu canvas');
                            setIsLoading(false);
                            return;
                        }

                        ctx.drawImage(img, 0, 0);

                        sourceCanvasRef.current = canvas;
                        originalDimsRef.current = { width: canvas.width, height: canvas.height };

                        URL.revokeObjectURL(objectUrl);
                        drawTiff(view);
                        setIsImageLoaded(true);
                        setIsLoading(false);
                    };

                    img.onerror = () => {
                        URL.revokeObjectURL(objectUrl);
                        if (isMounted) {
                            setError('Nie udało się wczytać obrazu JPG/PNG');
                            setIsLoading(false);
                        }
                    };

                    img.src = objectUrl;
                    return;
                }

                if (!window.Tiff) {
                    const msg = 'Tiff.js nie jest załadowany do obiektu window.';
                    if (isMounted) {
                        setError(msg);
                        setIsLoading(false);
                    }
                    return;
                }

                const response = await fetch(imageUrl);
                if (!response.ok) throw new Error(`Nie udało się pobrać TIFF (${response.status})`);

                const buffer = await response.arrayBuffer();
                if (!isMounted) return;

                const tiff = new window.Tiff({ buffer });
                tiff.setDirectory(0);

                const generatedCanvas = tiff.toCanvas();

                if (generatedCanvas) {
                    sourceCanvasRef.current = generatedCanvas;
                    originalDimsRef.current = {
                        width: generatedCanvas.width,
                        height: generatedCanvas.height,
                    };

                    drawTiff(view);

                    if (isMounted) {
                        setIsImageLoaded(true);
                        setError(null);
                    }
                } else {
                    if (isMounted) setError('Błąd dekodowania: tiff.toCanvas() zwróciło NULL/UNDEFINED.');
                }
            } catch (e: any) {
                if (isMounted) {
                    setError(e?.message ?? 'Nie udało się załadować obrazu.');
                }
            } finally {
                if (isMounted) setIsLoading(false);
            }
        };

        load();

        return () => {
            isMounted = false;
        };
    }, [imageUrl, drawTiff]);

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
        isDragging.current = false;
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

    return (
        <div className="image-viewer-wrapper">
            {error && <div className="error-message">BŁĄD: {error}</div>}

            <div className={`version-label ${activeVersion ? '' : 'hidden'}`}>
                Wersja: <strong>{activeVersion?.label ?? 'Brak'}</strong> ({versions.length > 0 ? activeIndex + 1 : 0}/{versions.length})
            </div>

            <div className="viewer-stage">
                {versions.length > 1 && (
                    <button className="viewer-nav viewer-nav-left" onClick={goPrev} aria-label="Poprzednia wersja">
                        {'<'}
                    </button>
                )}

                <div className="canvas-container" ref={containerRef}>
                    <canvas
                        ref={canvasRef}
                        className="tiff-canvas"
                        style={{
                            display: (error || !isImageLoaded) ? 'none' : 'block',
                            cursor: isImageLoaded ? (isDragging.current ? 'grabbing' : 'grab') : 'default',
                        }}
                    />

                    {isImageLoaded && renderGlomeruliOverlay()}

                    {isImageLoaded && activeVersion?.id === 'glomeruli' && (
                        <div className="conf-slider-overlay">
                            {!advancedOpen ? (
                                /* Tryb prosty */
                                <div className="conf-slider-row">
                                    <span>Próg ufności:</span>
                                    {/* min = conf modelu z backendu (slideInfo.conf).
                                        Poniżej tego progu model nie zwraca detekcji,
                                        więc suwak niżej nic nie zmienia. Fallback 15
                                        tylko gdy slideInfo jeszcze nie dotarło. */}
                                    <input
                                        type="range"
                                        min={Math.round((slideInfo?.conf ?? 0.15) * 100)}
                                        max={100}
                                        value={Math.round(confThresholds[0] * 100)}
                                        onChange={e => {
                                            const v = Number(e.target.value) / 100;
                                            onConfThresholdsChange?.({ 0: v, 1: v });
                                        }}
                                        style={{ width: 120, cursor: 'pointer' }}
                                    />
                                    <span style={{ minWidth: 36, textAlign: 'right' }}>
                                        {Math.round(confThresholds[0] * 100)}%
                                    </span>
                                    <button className="conf-advanced-toggle" onClick={() => setAdvancedOpen(true)}>
                                        Zaawansowane ▼
                                    </button>
                                    <button className="conf-advanced-toggle" onClick={() => setShowTiles(t => !t)}>
                                        {showTiles ? 'Ukryj kafelki' : 'Pokaż kafelki'}
                                    </button>
                                </div>
                            ) : (
                                /* Tryb zaawansowany */
                                <div className="conf-advanced">
                                    <div className="conf-slider-row">
                                        <span className="conf-label-healthy">Niezwłókniony:</span>
                                        <input
                                            type="range"
                                            min={Math.round((slideInfo?.conf ?? 0.15) * 100)}
                                            max={100}
                                            value={Math.round(confThresholds[0] * 100)}
                                            onChange={e => onConfThresholdsChange?.({ ...confThresholds, 0: Number(e.target.value) / 100 })}
                                            style={{ width: 100, cursor: 'pointer' }}
                                        />
                                        <span style={{ minWidth: 36, textAlign: 'right' }}>
                                            {Math.round(confThresholds[0] * 100)}%
                                        </span>
                                    </div>
                                    <div className="conf-slider-row">
                                        <span className="conf-label-sclerotic">Zwłókniony:</span>
                                        <input
                                            type="range"
                                            min={Math.round((slideInfo?.conf ?? 0.15) * 100)}
                                            max={100}
                                            value={Math.round(confThresholds[1] * 100)}
                                            onChange={e => onConfThresholdsChange?.({ ...confThresholds, 1: Number(e.target.value) / 100 })}
                                            style={{ width: 100, cursor: 'pointer' }}
                                        />
                                        <span style={{ minWidth: 36, textAlign: 'right' }}>
                                            {Math.round(confThresholds[1] * 100)}%
                                        </span>
                                    </div>
                                    <div className="conf-slider-row">
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
                                        <button className="conf-advanced-toggle" onClick={() => setShowTiles(t => !t)}>
                                            {showTiles ? 'Ukryj kafelki' : 'Pokaż kafelki'}
                                        </button>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    <div className={`scale-badge ${isImageLoaded && !error ? '' : 'hidden'}`}>
                        x{displayScale.toFixed(1)}
                    </div>
                </div>

                {versions.length > 1 && (
                    <button className="viewer-nav viewer-nav-right" onClick={goNext} aria-label="Następna wersja">
                        {'>'}
                    </button>
                )}
            </div>

            {(isLoading || !isImageLoaded) && !error && (
                <div className="status-message">
                    {isLoading ? 'Ładowanie obrazu...' : 'Oczekiwanie na dane obrazu...'}
                </div>
            )}
        </div>
    );
}