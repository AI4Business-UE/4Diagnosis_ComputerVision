import React, { useState, useEffect, useRef } from 'react';
import './AnnotationOverlay.css';
import type { GlomeruliDetection } from '../../services/api';

interface AnnotationOverlayProps {
    view: { scale: number; panX: number; panY: number };
    detections: GlomeruliDetection[];
    onUpdateDetections: (newDetections: GlomeruliDetection[]) => void;
    editMode: boolean;
}

export default function AnnotationOverlay({ view, detections, onUpdateDetections, editMode }: AnnotationOverlayProps) {
    const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const [drawingRect, setDrawingRect] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null);
    const [dragState, setDragState] = useState<{ type: string; startX: number; startY: number; initialRect?: any } | null>(null);

    const svgRef = useRef<SVGSVGElement>(null);

    const screenToImage = (clientX: number, clientY: number) => {
        if (!svgRef.current) return { x: 0, y: 0 };
        const rect = svgRef.current.getBoundingClientRect();
        const mouseX = clientX - rect.left;
        const mouseY = clientY - rect.top;
        return {
            x: (mouseX - view.panX) / view.scale,
            y: (mouseY - view.panY) / view.scale,
        };
    };

    const handlePointerDown = (e: React.PointerEvent) => {
        if (!editMode) return;
        if (e.button !== 0) return; // tylko lewy klik

        // Jeśli kliknięto w tło (nie w rect ani uchwyt)
        const target = e.target as SVGElement;
        if (target.tagName === 'svg') {
            const { x, y } = screenToImage(e.clientX, e.clientY);
            setIsDrawing(true);
            setDrawingRect({ x1: x, y1: y, x2: x, y2: y });
            setSelectedIdx(null);
            e.preventDefault();
        }
    };

    const handleRectPointerDown = (e: React.PointerEvent, idx: number) => {
        if (!editMode) {
            return;
        }

        e.stopPropagation(); // Blokuje event rysowania nowego prostokąta na SVG
        setSelectedIdx(idx);

        const { x, y } = screenToImage(e.clientX, e.clientY);
        setDragState({
            type: 'move',
            startX: x,
            startY: y,
            initialRect: { ...detections[idx] }
        });
        e.currentTarget.setPointerCapture(e.pointerId);
    };

    const handleHandlePointerDown = (e: React.PointerEvent, idx: number, handleType: string) => {
        if (!editMode) return;
        e.stopPropagation();
        setSelectedIdx(idx);
        
        const { x, y } = screenToImage(e.clientX, e.clientY);
        setDragState({
            type: `resize-${handleType}`,
            startX: x,
            startY: y,
            initialRect: { ...detections[idx] }
        });
        e.currentTarget.setPointerCapture(e.pointerId);
    };

    const handlePointerMove = (e: React.PointerEvent) => {
        if (!editMode) return;
        const { x, y } = screenToImage(e.clientX, e.clientY);

        if (isDrawing && drawingRect) {
            setDrawingRect(prev => prev ? { ...prev, x2: x, y2: y } : null);
            return;
        }

        if (dragState && selectedIdx !== null) {
            const dx = x - dragState.startX;
            const dy = y - dragState.startY;
            const initial = dragState.initialRect;
            
            let newX1 = initial.x1;
            let newY1 = initial.y1;
            let newX2 = initial.x2;
            let newY2 = initial.y2;

            if (dragState.type === 'move') {
                newX1 += dx;
                newX2 += dx;
                newY1 += dy;
                newY2 += dy;
            } else if (dragState.type.startsWith('resize-')) {
                const type = dragState.type.split('-')[1];
                if (type.includes('w')) newX1 += dx;
                if (type.includes('e')) newX2 += dx;
                if (type.includes('n')) newY1 += dy;
                if (type.includes('s')) newY2 += dy;
            }

            // Normalizacja (x1 musi być mniejsze od x2)
            const normX1 = Math.min(newX1, newX2);
            const normX2 = Math.max(newX1, newX2);
            const normY1 = Math.min(newY1, newY2);
            const normY2 = Math.max(newY1, newY2);

            const newDetections = [...detections];
            newDetections[selectedIdx] = {
                ...initial,
                x1: normX1,
                y1: normY1,
                x2: normX2,
                y2: normY2
            };
            onUpdateDetections(newDetections);
        }
    };

    const handlePointerUp = (e: React.PointerEvent) => {
        if (!editMode) return;

        if (isDrawing && drawingRect) {
            const x1 = Math.min(drawingRect.x1, drawingRect.x2);
            const x2 = Math.max(drawingRect.x1, drawingRect.x2);
            const y1 = Math.min(drawingRect.y1, drawingRect.y2);
            const y2 = Math.max(drawingRect.y1, drawingRect.y2);
            
            const width = x2 - x1;
            const height = y2 - y1;

            if (width > 10 && height > 10) {
                const newDet: GlomeruliDetection = {
                    x1, y1, x2, y2,
                    conf: 1.0,
                    cls: 0,
                    source: 'manual',
                    note: ''
                };
                const newDetections = [...detections, newDet];
                onUpdateDetections(newDetections);
                setSelectedIdx(newDetections.length - 1);
            }
        }

        setIsDrawing(false);
        setDrawingRect(null);
        setDragState(null);
        e.currentTarget.releasePointerCapture(e.pointerId);
    };

    // Usuwanie po wciśnięciu Delete
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (!editMode || selectedIdx === null) return;
            if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

            if (e.key === 'Delete' || e.key === 'Backspace') {
                const newDetections = detections.filter((_, i) => i !== selectedIdx);
                onUpdateDetections(newDetections);
                setSelectedIdx(null);
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [editMode, selectedIdx, detections, onUpdateDetections]);

    // Wyliczanie kursora dla kontenera css
    let containerClass = "annotation-overlay-container";
    if (editMode) containerClass += " edit-mode";
    if (dragState) {
        if (dragState.type === 'move') containerClass += " moving";
        else containerClass += ` resizing-${dragState.type.split('-')[1]}`;
    }

    // Renderowanie pojedynczego prostokąta
    const renderAnnotation = (det: GlomeruliDetection, i: number) => {
        const isSelected = selectedIdx === i;
        const rectProps = {
            x: det.x1 * view.scale + view.panX,
            y: det.y1 * view.scale + view.panY,
            width: (det.x2 - det.x1) * view.scale,
            height: (det.y2 - det.y1) * view.scale,
        };

        const handleRadius = 4;
        const handles = [
            { type: 'nw', x: rectProps.x, y: rectProps.y },
            { type: 'n', x: rectProps.x + rectProps.width / 2, y: rectProps.y },
            { type: 'ne', x: rectProps.x + rectProps.width, y: rectProps.y },
            { type: 'e', x: rectProps.x + rectProps.width, y: rectProps.y + rectProps.height / 2 },
            { type: 'se', x: rectProps.x + rectProps.width, y: rectProps.y + rectProps.height },
            { type: 's', x: rectProps.x + rectProps.width / 2, y: rectProps.y + rectProps.height },
            { type: 'sw', x: rectProps.x, y: rectProps.y + rectProps.height },
            { type: 'w', x: rectProps.x, y: rectProps.y + rectProps.height / 2 },
        ];

        const sourceClass = det.source === 'manual' ? 'source-manual' : 'source-ai';
        const labelText = det.source === 'manual' ? (det.note ? 'Manual 📝' : 'Manual') : `${(det.conf * 100).toFixed(0)}%`;

        return (
            <g key={i} className="annotation-group">
                {det.source === 'manual' && editMode ? (
                    <foreignObject x={rectProps.x} y={rectProps.y - 28} width={150} height={26}>
                        <input
                            type="text"
                            value={det.note || ''}
                            onChange={(e) => {
                                const newDetections = [...detections];
                                newDetections[i].note = e.target.value;
                                onUpdateDetections(newDetections);
                            }}
                            onPointerDown={(e) => e.stopPropagation()}
                            placeholder="Notatka..."
                            style={{
                                width: '100%',
                                height: '100%',
                                background: 'rgba(0, 0, 0, 0.7)',
                                border: '1px solid #ffaa00',
                                color: 'white',
                                padding: '0 4px',
                                fontSize: '12px',
                                borderRadius: '4px',
                                outline: 'none',
                                boxSizing: 'border-box'
                            }}
                        />
                    </foreignObject>
                ) : (
                    <>
                        <rect 
                            className="annotation-label-bg"
                            x={rectProps.x} 
                            y={rectProps.y - 18} 
                            width={det.source === 'manual' ? (det.note ? 75 : 55) : 35} 
                            height={16} 
                            rx={2}
                        />
                        <text 
                            className="annotation-label-text"
                            x={rectProps.x + 4} 
                            y={rectProps.y - 6}
                        >
                            {labelText}
                        </text>
                    </>
                )}

                <rect
                    className={`annotation-rect ${sourceClass} ${isSelected ? 'selected' : ''}`}
                    x={rectProps.x}
                    y={rectProps.y}
                    width={rectProps.width}
                    height={rectProps.height}
                    onPointerDown={(e) => handleRectPointerDown(e, i)}
                    style={{ cursor: editMode ? 'move' : 'pointer' }}
                />

                {editMode && isSelected && handles.map(h => (
                    <circle
                        key={h.type}
                        className="resize-handle"
                        cx={h.x}
                        cy={h.y}
                        r={handleRadius}
                        onPointerDown={(e) => handleHandlePointerDown(e, i, h.type)}
                        style={{ cursor: `${h.type}-resize` }}
                    />
                ))}
            </g>
        );
    };

    return (
        <div className={containerClass}>
            {editMode && (
                <div className="edit-mode-banner">
                    ✏️ Tryb edycji: Rysuj kwadraty myszką, kliknij by zaznaczyć, użyj Delete by usunąć.
                </div>
            )}
            <div className="annotation-legend">
                <h4>Legenda detekcji:</h4>
                <div className="legend-item">
                    <span className="legend-color ai-color"></span>
                    <span>Model AI (% = pewność)</span>
                </div>
                <div className="legend-item">
                    <span className="legend-color manual-color"></span>
                    <span>Ręczne (Manual)</span>
                </div>
            </div>
            <svg 
                ref={svgRef}
                className="annotation-svg"
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
                onPointerLeave={handlePointerUp}
            >
                {detections.map((det, i) => renderAnnotation(det, i))}

                {isDrawing && drawingRect && (
                    <rect
                        className="annotation-rect source-manual"
                        x={Math.min(drawingRect.x1, drawingRect.x2) * view.scale + view.panX}
                        y={Math.min(drawingRect.y1, drawingRect.y2) * view.scale + view.panY}
                        width={Math.abs(drawingRect.x2 - drawingRect.x1) * view.scale}
                        height={Math.abs(drawingRect.y2 - drawingRect.y1) * view.scale}
                        strokeDasharray="4"
                    />
                )}
            </svg>
        </div>
    );
}
