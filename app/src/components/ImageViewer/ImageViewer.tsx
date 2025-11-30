import { useRef, useEffect, useState } from 'react';
import './ImageViewer.css';

import exampleTiffUrl from '../../assets/example.tiff'; 

export default function ImageViewer() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const load = async () => {
            if (!window.Tiff) {
                setError("Tiff.js nie jest załadowany");
                console.log(error);
                return;
            }

            setIsLoading(true);

            const response = await fetch(exampleTiffUrl);
            const buffer = await response.arrayBuffer();

            const tiff = new window.Tiff({ buffer });

            tiff.setDirectory(0);
            const canvas = tiff.toCanvas();

            if (canvasRef.current && canvas) {
                canvasRef.current.width = canvas.width;
                canvasRef.current.height = canvas.height;
                const ctx = canvasRef.current.getContext("2d");
                ctx?.drawImage(canvas, 0, 0);
            }

            setIsLoading(false);
        };

        load();
    }, []);

    return (
        <div className="image-container">
            {isLoading
                ?   <div className="status-message">Ładowanie obrazu TIFF...</div>
                :   <div className="canvas-container">
                        <canvas ref={canvasRef} className="tiff-canvas"></canvas>
                    </div>}
        </div>
    );
}