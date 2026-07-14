import './ControlPanel.css'
import { useState } from 'react'
import { useNotification } from '../Notifications/NotificationContext'
import LoadingScreen from '../LoadingScreen/LoadingScreen'
import { selectFolder } from '../../services/api'
import { runConvert, runFibrosis, runLength, runGlomeruli } from '../../services/pipeline'
import type { Glomeruli, SlideInfo, TileInfo } from '../../services/api'
import type { Sample } from '../../types/Sample'

interface ControlPanelProps {
    activeSample: Sample | null;
    /** True while the "Analizuj wszystko" batch owns the pipeline. */
    batchRunning?: boolean;
    onSamplesDetected: (samples: Array<{ name: string; files: File[] }>) => void;
    onTiffReady?: (tiffUrl: string | null) => void;
    onOverlayReady?: (id: 'fibrosis' | 'length' | 'glomeruli' | 'glom_grid', label: string, url: string) => void;
    onAnalysisComplete?: (data: Sample['analysisResult']) => void;
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
            }
        } else if (pathParts.length >= 3) {
            const folderName = pathParts[1];

            if (!sampleMap.has(folderName)) {
                sampleMap.set(folderName, []);
            }
            sampleMap.get(folderName)!.push(file);
        }
    }

    return Array.from(sampleMap.entries()).map(([name, files]) => ({ name, files }));
}

export default function ControlPanel({
        activeSample,
        batchRunning = false,
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

    const busy = batchRunning || isLoading || isAnalyzing !== null;
    const notConverted = !activeSample || activeSample.processStage !== 'converted';

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
            const { jobId, previewUrl } = await runConvert(activeSample.files);

            onJobIdChange?.(jobId);
            onStageChange?.('converted');
            onTiffReady?.(previewUrl);

            addNotification(`Konwersja zakończona: ${activeSample.name}`, 'success');
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
        if (isAnalyzing) return;

        setIsAnalyzing('fibrosis');
        const loadingId = addNotification('Analiza zwłóknienia...', 'loading');

        try {
            const { data, overlayUrl } = await runFibrosis(activeSample.jobId);

            onAnalysisComplete?.({
                fibrosis_ratio: data.fibrosis_ratio,
                fibrosis_ratio_avg: data.fibrosis_ratio_avg,
                fibrosis_warning: data.fibrosis_warning ?? false,
            });

            // Slices disagree — the representative value may not be trustworthy.
            if (data.fibrosis_warning) {
                addNotification(
                    `⚠️ Uwaga: wyniki zwłóknienia różnią się między slicami. Avg: ${data.fibrosis_ratio_avg != null ? (data.fibrosis_ratio_avg * 100).toFixed(1) : '?'}%, reprezentant: ${data.fibrosis_ratio != null ? (data.fibrosis_ratio * 100).toFixed(1) : '?'}%`,
                    'error',
                    8000
                );
            }

            if (overlayUrl) onOverlayReady?.('fibrosis', 'Zwłóknienie (overlay)', overlayUrl);

            onAnalysisStatusChange?.('fibrosis', true);
            addNotification('Analiza zwłóknienia zakończona', 'success');
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
        if (isAnalyzing) return;

        setIsAnalyzing('length');
        const loadingId = addNotification('Analizowanie długości...', 'loading');

        try {
            const { data, overlayUrl } = await runLength(activeSample.jobId);

            onAnalysisComplete?.({ length: data.length });

            if (overlayUrl) onOverlayReady?.('length', 'Długość tkanki (overlay)', overlayUrl);

            onAnalysisStatusChange?.('length', true);
            addNotification('Analiza długości zakończona!', 'success');
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Błąd podczas analizy długości';
            addNotification(message, 'error', 5000);
        } finally {
            removeNotification(loadingId);
            setIsAnalyzing(null);
        }
    };

    const handleGlomeruli = async () => {
        if (!activeSample?.jobId) {
            addNotification('Najpierw wykonaj konwersję', 'error');
            return;
        }
        if (isAnalyzing) return;

        onGlomeruliReset?.();
        setIsAnalyzing('glomeruli');
        onGlomeruliScanning?.(true);
        const loadingId = addNotification('Wykrywanie kłębuszków...', 'loading');

        try {
            const result = await runGlomeruli(activeSample.jobId, {
                onSlideInfo: (info) => onSlideInfo?.(info),
                onGlomeruli: (batch) => onGlomeruliDetected?.(batch),
                onTiles: (tiles) => onTilesUpdate?.(tiles),
            });

            onFinalGlomeruliList?.(result.glomeruli);
            onAnalysisComplete?.({ glomeruli_count: result.count });
            onAnalysisStatusChange?.('glomeruli', true);

            if (result.gridUrl) {
                onOverlayReady?.('glom_grid', 'Porównanie kłębuszków (siatka)', result.gridUrl);
            }

            addNotification(`Wykryto ${result.count} kłębuszków`, 'success');
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Błąd wykrywania kłębuszków';
            addNotification(message, 'error', 5000);
        } finally {
            removeNotification(loadingId);
            setIsAnalyzing(null);
            onGlomeruliScanning?.(false);
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
                    disabled={busy}
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
                    disabled={busy || !activeSample || activeSample.processStage !== 'folder_selected'}
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
                        disabled={busy || notConverted}
                        id="fibrosis"
                        onClick={handleFibrosis}
                        className={activeSample?.fibrosisCompleted ? 'completed' : ''}
                    >
                        <img src="/analyze.svg" width={20} height={20} alt="" />
                        <span>Analizuj zwłóknienie</span>
                        {activeSample?.fibrosisCompleted && <span className="checkmark">✓</span>}
                    </button>

                    <button
                        disabled={busy || notConverted}
                        id="length"
                        onClick={handleLength}
                        className={activeSample?.lengthCompleted ? 'completed' : ''}
                    >
                        <img src="/ruler.svg" width={20} height={20} alt="" />
                        <span>Analizuj długość</span>
                        {activeSample?.lengthCompleted && <span className="checkmark">✓</span>}
                    </button>

                    <button
                        disabled={busy || notConverted}
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
