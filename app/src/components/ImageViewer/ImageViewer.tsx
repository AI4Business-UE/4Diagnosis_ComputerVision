import { useRef, useEffect, useState, useCallback } from 'react';
import './ImageViewer.css';

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
    directory: FileSystemDirectoryHandle | null;
}

const BASE_SCALE = 0.2;

export default function ImageViewer({ directory }: ImageViewerProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [isImageLoaded, setIsImageLoaded] = useState(false);
    
    const sourceCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const originalDimsRef = useRef<{ width: number, height: number } | null>(null);
    
    const [view, setView] = useState<ViewState>({
        scale: BASE_SCALE,
        panX: 0,
        panY: 0,
    });
    
    const isDragging = useRef(false);
    const lastMousePos = useRef({ x: 0, y: 0 });

    const drawTiff = useCallback((currentView: ViewState) => {
        const canvas = canvasRef.current;
        const sourceCanvas = sourceCanvasRef.current;
        const dims = originalDimsRef.current;

        if (!canvas || !sourceCanvas || !dims) {
            return;
        }

        const ctx = canvas.getContext("2d");
        if (ctx) {
            canvas.width = dims.width;
            canvas.height = dims.height;

            ctx.clearRect(0, 0, dims.width, dims.height);
            ctx.save();
            
            ctx.translate(currentView.panX, currentView.panY);
            ctx.scale(currentView.scale, currentView.scale); 
            
            ctx.drawImage(sourceCanvas, 0, 0); 
            
            ctx.restore();
        }
    }, []);


    useEffect(() => {
        let isMounted = true; 

        const load = async () => {
            if (!isMounted) return;
            
            if (!directory) {
                if (isMounted) {
                    setError('Żaden folder nie został wybrany. Wybierz folder w panelu sterowania.');
                    setIsLoading(false);
                }
                return;
            }
            
            if (!window.Tiff) {
                const msg = "Tiff.js nie jest załadowany do obiektu window.";
                if (isMounted) {
                    setError(msg);
                    setIsLoading(false);
                }
                return;
            }

            if (!isMounted) return;

            try {
                let tiffFile: File | null = null;

                for await (const handle of (directory as any).values()) {
                    if (handle.kind === 'file' && (handle.name.endsWith('.tiff') || handle.name.endsWith('.tif'))) {
                        tiffFile = await (handle as FileSystemFileHandle).getFile();
                        break;
                    }
                }

                if (!tiffFile) {
                    if (isMounted) {
                        setError('Nie znaleziono pliku TIFF w wybranym folderze.');
                        setIsLoading(false);
                    }
                    return;
                }

                if (!isMounted) return;

                const buffer = await tiffFile.arrayBuffer();

                if (!isMounted) return;

                const tiff = new window.Tiff({ buffer });
                tiff.setDirectory(0);
                
                const generatedCanvas = tiff.toCanvas(); 

                if (generatedCanvas) {
                    sourceCanvasRef.current = generatedCanvas;
                    originalDimsRef.current = { 
                        width: generatedCanvas.width, 
                        height: generatedCanvas.height 
                    };
                    
                    drawTiff(view);
                    
                    if (isMounted) {
                        setIsImageLoaded(true);
                        setError(null);
                    }
                } else {
                    const msg = 'Błąd dekodowania: tiff.toCanvas() zwróciło NULL/UNDEFINED.';
                    if (isMounted) setError(msg);
                }
                
            } catch (e: any) {
                if (isMounted) {
                    setError(e?.message ?? 'Nie udało się załadować TIFF.');
                }
            } finally {
                if (isMounted) {
                    setIsLoading(false);
                }
            }
        };

        load();

        return () => {
            isMounted = false;
        };
    }, [directory, view, drawTiff]);

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
        const maxScale = 4.0; 
        const minScale = 0.2;

        const newScale = Math.max(minScale, Math.min(view.scale + zoomFactor, maxScale));
        
        if (newScale === view.scale) return;
        
        const rect = canvas.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;

        const oldImageX = (mouseX - view.panX) / view.scale;
        const oldImageY = (mouseY - view.panY) / view.scale;
        
        const newPanX = mouseX - oldImageX * newScale;
        const newPanY = mouseY - oldImageY * newScale;

        setView(prevView => ({
            scale: newScale,
            panX: newPanX,
            panY: newPanY,
        }));

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
            
            window.addEventListener('mousedown', handleMouseDown);
            window.addEventListener('mousemove', handleMouseMove);
            window.addEventListener('mouseup', handleMouseUp);
        }

        return () => {
            if (canvas) {
                canvas.removeEventListener('wheel', handleWheel);
            }
            window.removeEventListener('mousedown', handleMouseDown);
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        };
    }, [handleWheel, handleMouseDown, handleMouseMove, handleMouseUp]);
    
    const displayScale = view.scale / BASE_SCALE;


    return (
        <div className="image-viewer-wrapper">
            {error && <div className="error-message">BŁĄD: {error}</div>}
            
            <div className={`controls ${isImageLoaded && !error ? '' : 'hidden'}`}>
                <span>Skala: x{displayScale.toFixed(1)}</span>
            </div>

            <div className="canvas-container">
                <canvas 
                    ref={canvasRef} 
                    className="tiff-canvas"
                    style={{ 
                        display: (error || !isImageLoaded) ? 'none' : 'block',
                        cursor: isImageLoaded ? (isDragging.current ? 'grabbing' : 'grab') : 'default',
                    }}
                />
            </div>

            {(isLoading || !isImageLoaded) && !error && (
                <div className="status-message">
                    {isLoading ? 'Ładowanie obrazu TIFF...' : 'Oczekiwanie na dane obrazu...'}
                </div>
            )}
        </div>
    );
}