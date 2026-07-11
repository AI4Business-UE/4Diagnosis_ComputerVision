import './ControlPanel.css'
import { useState } from 'react'
import { useNotification } from '../Notifications/NotificationContext'
import LoadingScreen from '../LoadingScreen/LoadingScreen'
import {
    selectFolder,
    convertToTiff,
    analyzeFibrosis,
    analyzeLength,
} from '../../services/api'
import type { Glomeruli, SlideInfo, TileInfo } from '../../services/api'
import type { Sample } from '../../types/Sample'

const API_ORIGIN = 'http://localhost:8000';

interface ControlPanelProps {
    activeSample: Sample | null;
    onSamplesDetected: (samples: Array<{ name: string; files: File[] }>) => void;
    onTiffReady?: (tiffUrl: string | null) => void;
    onOverlayReady?: (id: 'fibrosis' | 'length' | 'glomeruli' | 'glom_grid', label: string, url: string) => void;
    onAnalysisComplete?: (data: any) => void;
    onStageChange?: (stage: Sample['processStage']) => void;
    onJobIdChange?: (jobId: string) => void;
    onAnalysisStatusChange?: (type: 'fibrosis' | 'length' | 'glomeruli', completed: boolean) => void;
    onGlomeruliScanning?: (scanning: boolean) => void;
    onGlomeruliDetected?: (batch: Glomeruli[]) => void;
    onSlideInfo?: (info: SlideInfo) => void;
    onGlomeruliReset?: () => void;
    onTilesUpdate?: (tiles: TileInfo[]) => void;
    onFinalGlomeruliList?: (list: Glomeruli[]) => void;
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
                console.log(`Znaleziono plik .mrxs dla próbki: ${sampleName}`);
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
        const mrxsFiles = files.filter(f => f.name.endsWith('.mrxs'));
        const otherFiles = files.filter(f => !f.name.endsWith('.mrxs'));
        
        console.log(`Próbka "${name}":`);
        console.log(`  - plików .mrxs: ${mrxsFiles.length}`);
        console.log(`  - innych plików: ${otherFiles.length}`);
        console.log(`  - razem: ${files.length}`);
        
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
        onAnalysisStatusChange,
        onGlomeruliScanning,
        onGlomeruliDetected,
        onSlideInfo,
        onGlomeruliReset,
        onTilesUpdate,
        onFinalGlomeruliList,
    }: ControlPanelProps) {
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
                
                const previewUrl = data.origin_detect_url || data.tiff_url;
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
        console.log('Analiza zwłóknienia już trwa, ignoruję kliknięcie');
        return;
    }

    setIsAnalyzing('fibrosis');
    const loadingId = addNotification('Analiza zwłóknienia...', 'loading');

    try {
        const result = await analyzeFibrosis(activeSample.jobId);

        if (result.success && result.data) {
            const fibrosisData = {
                fibrosis_ratio: result.data.fibrosis_ratio,
                fibrosis_ratio_avg: result.data.fibrosis_ratio_avg,
                fibrosis_warning: result.data.fibrosis_warning ?? false,
                fibrotic_pixels: result.data.fibrotic_pixels,
                tissue_pixels: result.data.tissue_pixels,
            };

            onAnalysisComplete?.(fibrosisData);

            // Fibrosis warning — significant difference between slices
            if (result.data.fibrosis_warning) {
                addNotification(
                    `⚠️ Uwaga: wyniki zwłóknienia różnią się między slicami. Avg: ${result.data.fibrosis_ratio_avg != null ? (result.data.fibrosis_ratio_avg * 100).toFixed(1) : '?'}%, reprezentant: ${result.data.fibrosis_ratio != null ? (result.data.fibrosis_ratio * 100).toFixed(1) : '?'}%`,
                    'error',
                    8000
                );
            }

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
        console.log('Analiza długości już trwa, ignoruję kliknięcie');
        return;
    }

    setIsAnalyzing('length');
    const loadingId = addNotification('Analizowanie długości...', 'loading');

    try {
        const result = await analyzeLength(activeSample.jobId);

        if (result.success && result.data) {
            // Przekaż TYLKO dane długości
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

    const handleGlomeruli = () => {
        if (!activeSample?.jobId) {
            addNotification('Najpierw wykonaj konwersję', 'error');
            return;
        }
        if (isAnalyzing === 'glomeruli') return;

        onGlomeruliReset?.();
        setIsAnalyzing('glomeruli');
        onGlomeruliScanning?.(true);
        const loadingId = addNotification('Wykrywanie kłębuszków...', 'loading');

        const url = `${API_ORIGIN}/api/glomeruli/stream/?job_id=${encodeURIComponent(activeSample.jobId)}`;
        const es = new EventSource(url);

        es.onmessage = (e) => {
            try {
                const msg = JSON.parse(e.data);
                if (msg.error) {
                    addNotification(msg.error, 'error', 5000);
                    es.close();
                    removeNotification(loadingId);
                    setIsAnalyzing(null);
                    onGlomeruliScanning?.(false);
                    return;
                }
                if (msg.slide_info) {
                    onSlideInfo?.(msg.slide_info);
                }
                if (msg.glomeruli) {
                    onGlomeruliDetected?.(msg.glomeruli);
                }
                if (msg.tiles) {
                    onTilesUpdate?.(msg.tiles);
                }
                if (msg.done) {
                    onFinalGlomeruliList?.(msg.final_glomeruli ?? []);
                    onAnalysisComplete?.({ glomeruli_count: msg.count ?? 0 });
                    onAnalysisStatusChange?.('glomeruli', true);
                    onGlomeruliScanning?.(false);

                    // all-slices mode: show comparison grid if available
                    if (msg.glom_grid_url && activeSample?.jobId) {
                        const gridUrl = `${API_ORIGIN}${msg.glom_grid_url}`;
                        onOverlayReady?.('glom_grid', 'Porównanie kłębuszków (siatka)', gridUrl);
                    }

                    addNotification(`Wykryto ${msg.count ?? 0} kłębuszków`, 'success');
                    es.close();
                    removeNotification(loadingId);
                    setIsAnalyzing(null);
                }
            } catch { /* ignore parse errors */ }
        };

        es.onerror = () => {
            addNotification('Błąd połączenia SSE', 'error', 5000);
            es.close();
            removeNotification(loadingId);
            setIsAnalyzing(null);
            onGlomeruliScanning?.(false);
        };
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
                        id="glomeruli"
                        onClick={handleGlomeruli}
                        className={activeSample?.glomeruliCompleted ? 'completed' : ''}
                    >
                        <img src="/detect.svg" width={20} height={20} alt="" />
                        <span>Wykryj kłębuszki</span>
                        {activeSample?.glomeruliCompleted && <span className="checkmark">✓</span>}
                    </button>
                </div>
            </div>
        </>
    );
}