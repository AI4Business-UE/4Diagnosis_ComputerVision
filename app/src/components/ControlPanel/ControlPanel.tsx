import './ControlPanel.css'
import { useState } from 'react'
import { useNotification } from '../Notifications/NotificationContext'
import LoadingScreen from '../LoadingScreen/LoadingScreen'
import {
    selectFolder,
    convertToTiff,
    analyzeFibrosis,
    analyzeLength,
    detectGlomerules,
} from '../../services/api'
import type { Sample } from '../../types/Sample'

const API_ORIGIN = 'http://127.0.0.1:8000';

interface ControlPanelProps {
    activeSample: Sample | null;
    onSamplesDetected: (samples: Array<{ name: string; files: File[] }>) => void;
    onTiffReady?: (tiffUrl: string | null) => void;
    onOverlayReady?: (id: 'fibrosis' | 'length' | 'glomeruli', label: string, url: string) => void;
    onAnalysisComplete?: (data: any) => void;
    onStageChange?: (stage: Sample['processStage']) => void;
    onJobIdChange?: (jobId: string) => void;
    onAnalysisStatusChange?: (type: 'fibrosis' | 'length' | 'glomerules', completed: boolean) => void;
}

function detectSamplesFromFiles(files: File[]): Array<{ name: string; files: File[] }> {

    const sampleMap = new Map<string, File[]>();

    for (const file of files) {
        const pathParts = file.webkitRelativePath.split('/');
        
        if (pathParts.length === 2) {
            const fileName = pathParts[1];
            if (fileName.endsWith('.mrxs')) {
                const sampleName = fileName.replace('.mrxs', '');
                
                if (!sampleMap.has(sampleName)) {
                    sampleMap.set(sampleName, []);
                }
                sampleMap.get(sampleName)!.push(file);
            }
        } else if (pathParts.length >= 3) {
            const folderName = pathParts[1];
            
            if (!sampleMap.has(folderName)) {
                sampleMap.set(folderName, []);
            }
            sampleMap.get(folderName)!.push(file);
        }
    }

    const result = Array.from(sampleMap.entries()).map(([name, files]) => {
        return { name, files };
    });
    return result;
}

export default function ControlPanel({
        activeSample,
        onSamplesDetected,
        onTiffReady,
        onOverlayReady,
        onAnalysisComplete,
        onStageChange,
        onJobIdChange,
        onAnalysisStatusChange
    }   : ControlPanelProps) {
    const [isLoading, setIsLoading] = useState(false);
    const [isAnalyzing, setIsAnalyzing] = useState<'fibrosis' | 'length' | 'glomeruli' | null>(null);
    const { addNotification, removeNotification } = useNotification();

    const toResultImageUrl = (imagePath: string, jobId: string | null) => {
        if (!jobId) return null;
        const fileName = imagePath.split(/[/\\]/).pop();
        if (!fileName) return null;
        return `${API_ORIGIN}/api/result-image/${encodeURIComponent(jobId)}/${encodeURIComponent(fileName)}/`;
    };

    const handleSelectFolder = () => {
        const input = document.getElementById('folder-input') as HTMLInputElement | null;
        if (!input) return;
        input.value = '';
        input.click();
    };

    const handleFolderUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!e.target.files || e.target.files.length === 0) return;

        const files = Array.from(e.target.files);
        const detectedSamples = detectSamplesFromFiles(files);

        if (detectedSamples.length === 0) {
            addNotification('Nie znaleziono próbek w wybranym folderze', 'error');
            return;
        }

        addNotification(`Znaleziono ${detectedSamples.length} próbek`, 'success', 2000);

        onSamplesDetected(detectedSamples);

        try {
            await selectFolder('_initial_clear');
        } catch (error) {
            console.error('Błąd podczas czyszczenia cache:', error);
        }
    };

    const handleConvert = async () => {
        if (!activeSample || activeSample.files.length === 0) {
            addNotification('Wybierz próbkę przed konwersją', 'error');
            return;
        }

        setIsLoading(true);
        const loadingId = addNotification(`Konwersja: ${activeSample.name}...`, 'loading');

        try {
            
            const result = await convertToTiff(activeSample.files);

            if (!result.success || !result.data) {
                throw new Error(result.error || 'Nieznany błąd konwersji');
            }

            const data = result.data;
            if (data.job_id) {
                onJobIdChange?.(data.job_id);
                onStageChange?.('converted');
                
                const previewUrl = data.mask_preview_url || data.tiff_url;
                const fullPreviewUrl = previewUrl ? `${API_ORIGIN}${previewUrl}` : null;
                onTiffReady?.(fullPreviewUrl);
                
                addNotification(`Konwersja zakończona: ${activeSample.name}`, 'success');
            } else {
                throw new Error('Backend nie zwrócił job_id');
            }

            onAnalysisComplete?.(data);
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Błąd konwersji';
            addNotification(message, 'error', 5000);
        } finally {
            removeNotification(loadingId);
            setIsLoading(false);
        }
    };

    const handleFibrosis = async () => {
    if (!activeSample?.jobId) {
        addNotification('Najpierw wykonaj konwersję', 'error');
        return;
    }

    if (isAnalyzing === 'fibrosis') {
        return; // Already running
    }

    setIsAnalyzing('fibrosis');
    const loadingId = addNotification('Analiza zwłóknienia...', 'loading');

    try {
        const result = await analyzeFibrosis(activeSample.jobId);

        if (result.success && result.data) {
            // Pass ONLY fibrosis data
            const fibrosisData = {
                fibrosis_ratio: result.data.fibrosis_ratio,
                fibrotic_pixels: result.data.fibrotic_pixels,
                tissue_pixels: result.data.tissue_pixels,
            };
            
            onAnalysisComplete?.(fibrosisData);

            if (typeof result.data.image_path === 'string' && result.data.image_path.length > 0) {
                const overlayUrl = toResultImageUrl(result.data.image_path, activeSample.jobId);
                if (overlayUrl) {
                    onOverlayReady?.('fibrosis', 'Zwłóknienie (overlay)', overlayUrl);
                }
            }

            onAnalysisStatusChange?.('fibrosis', true);
            addNotification('Analiza zwłóknienia zakończona', 'success');
        } else {
            throw new Error(result.error || 'Błąd analizy zwłóknienia');
        }
    } catch (err) {
        const message = err instanceof Error ? err.message : 'Błąd analizy zwłóknienia';
        addNotification(message, 'error', 5000);
    } finally {
        removeNotification(loadingId);
        setIsAnalyzing(null);
    }
};

   const handleLength = async () => {
    if (!activeSample?.jobId) {
        addNotification('Najpierw wykonaj konwersję', 'error');
        return;
    }

    if (isAnalyzing === 'length') {
        return; // Already running
    }

    setIsAnalyzing('length');
    const loadingId = addNotification('Analizowanie długości...', 'loading');

    try {
        const result = await analyzeLength(activeSample.jobId);

        if (result.success && result.data) {
            // Pass ONLY length data
            const lengthData = {
                length: result.data.length,
            };
            
            onAnalysisComplete?.(lengthData);

            if (typeof result.data.image_path === 'string' && result.data.image_path.length > 0) {
                const overlayUrl = toResultImageUrl(result.data.image_path, activeSample.jobId);
                if (overlayUrl) {
                    onOverlayReady?.('length', 'Długość tkanki (overlay)', overlayUrl);
                }
            }

            onAnalysisStatusChange?.('length', true);
            addNotification('Analiza długości zakończona!', 'success');
        } else {
            throw new Error(result.error || 'Błąd analizy długości');
        }
    } catch (err) {
        const message = err instanceof Error ? err.message : 'Błąd podczas analizy długości';
        addNotification(message, 'error', 5000);
    } finally {
        removeNotification(loadingId);
        setIsAnalyzing(null);
    }
};

    const handleGlomerule = async () => {
    if (!activeSample?.jobId) {
        addNotification('Najpierw wykonaj konwersję', 'error');
        return;
    }

    if (isAnalyzing === 'glomeruli') {
        return; // Already running
    }

    setIsAnalyzing('glomeruli');
    const loadingId = addNotification('Wykrywanie kłębuszków...', 'loading');

    try {
        const result = await detectGlomerules(activeSample.jobId);

        if (result.success && result.data) {
            // Pass ONLY glomeruli data
            const glomeruliData = {
                glomeruli_count: result.data.count ?? 0,
                detections: result.data.detections ?? [],
            };

            onAnalysisComplete?.(glomeruliData);

            if (typeof result.data.image_url === 'string' && result.data.image_url.length > 0) {
                const overlayUrl = `${API_ORIGIN}${result.data.image_url}`;
                onOverlayReady?.('glomeruli', `Kłębuszki (${result.data.count ?? 0})`, overlayUrl);
            }

            onAnalysisStatusChange?.('glomerules', true);
            addNotification('Wykrycie kłębuszków zakończone!', 'success');
        } else {
            throw new Error(result.error || 'Błąd wykrywania');
        }
    } catch (err) {
        const message = err instanceof Error ? err.message : 'Błąd podczas wykrywania kłębuszków';
        addNotification(message, 'error', 5000);
    } finally {
        removeNotification(loadingId);
        setIsAnalyzing(null);
    }
};

    return (
        <>
            <LoadingScreen isVisible={isLoading} progress={0} />

            <div className="control-panel">
                <p>Panel sterowania</p>
                
                <input
                    id="folder-input"
                    type="file"
                    multiple
                    style={{ display: 'none' }}
                    onChange={handleFolderUpload}
                    //@ts-ignore
                    webkitdirectory=""
                />

                <button
                    id="choose-folder"
                    onClick={handleSelectFolder}
                    className={activeSample ? 'selected' : ''}
                >
                    <img src="/folder-open.svg" width={20} height={20} alt="" />
                    <span>Wybierz folder z próbkami</span>
                    {activeSample && <span className="checkmark">✓</span>}
                </button>

                {activeSample && (
                    <div className="selected-folder">
                        <strong>Aktywna próbka:</strong> {activeSample.name}
                    </div>
                )}

                <button
                    disabled={!activeSample || activeSample.processStage !== 'folder_selected'}
                    id="convert"
                    onClick={handleConvert}
                    className={activeSample?.processStage === 'converted' ? 'completed' : ''}
                >
                    <img src="/convert.svg" width={20} height={20} alt="" />
                    <span>Konwertuj</span>
                    {activeSample?.processStage === 'converted' && <span className="checkmark">✓</span>}
                </button>

                <div className="analysis-section">
                    <p>Analiza</p>

                    <button
                        disabled={!activeSample || activeSample.processStage !== 'converted'}
                        id="fibrosis"
                        onClick={handleFibrosis}
                        className={activeSample?.fibrosisCompleted ? 'completed' : ''}
                    >
                        <img src="/analyze.svg" width={20} height={20} alt="" />
                        <span>Analizuj zwłóknienie</span>
                        {activeSample?.fibrosisCompleted && <span className="checkmark">✓</span>}
                    </button>

                    <button
                        disabled={!activeSample || activeSample.processStage !== 'converted'}
                        id="length"
                        onClick={handleLength}
                        className={activeSample?.lengthCompleted ? 'completed' : ''}
                    >
                        <img src="/analyze.svg" width={20} height={20} alt="" />
                        <span>Analizuj długość</span>
                        {activeSample?.lengthCompleted && <span className="checkmark">✓</span>}
                    </button>

                    <button
                        disabled={!activeSample || activeSample.processStage !== 'converted'}
                        id="glomerule"
                        onClick={handleGlomerule}
                        className={activeSample?.glomerulesCompleted ? 'completed' : ''}
                    >
                        <img src="/detect.svg" width={20} height={20} alt="" />
                        <span>Wykryj kłębuszki</span>
                        {activeSample?.glomerulesCompleted && <span className="checkmark">✓</span>}
                    </button>
                </div>
            </div>
        </>
    );
}