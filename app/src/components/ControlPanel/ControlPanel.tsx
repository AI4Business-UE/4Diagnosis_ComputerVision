import './ControlPanel.css'
import { useNotification } from '../Notifications/NotificationContext'
import { selectFolder } from '../../services/api'
import type { Sample } from '../../types/Sample'

type AnalysisKind = 'fibrosis' | 'length' | 'glomeruli';

interface ControlPanelProps {
    activeSample: Sample | null;
    /** True while the "Analizuj wszystko" batch owns the pipeline. */
    batchRunning?: boolean;
    isConverting?: boolean;
    runningAnalyses?: Set<AnalysisKind>;
    onSamplesDetected: (samples: Array<{ name: string; files: File[] }>) => void;
    onConvertSample?: (sample: Sample) => void;
    onRunFibrosis?: (sample: Sample) => void;
    onRunLength?: (sample: Sample) => void;
    onRunGlomeruli?: (sample: Sample) => void;
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
        isConverting = false,
        runningAnalyses,
        onSamplesDetected,
        onConvertSample,
        onRunFibrosis,
        onRunLength,
        onRunGlomeruli,
    }: ControlPanelProps) {
    const { addNotification } = useNotification();

    const isFibrosisRunning = Boolean(runningAnalyses?.has('fibrosis'));
    const isLengthRunning = Boolean(runningAnalyses?.has('length'));
    const isGlomeruliRunning = Boolean(runningAnalyses?.has('glomeruli'));
    const hasAnyAnalysis = isFibrosisRunning || isLengthRunning || isGlomeruliRunning;
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

    const handleConvert = () => {
        if (!activeSample || activeSample.files.length === 0) {
            addNotification('Wybierz próbkę przed konwersją', 'error');
            return;
        }
        onConvertSample?.(activeSample);
    };

    const handleFibrosis = () => {
        if (!activeSample?.jobId) {
            addNotification('Najpierw wykonaj konwersję', 'error');
            return;
        }
        onRunFibrosis?.(activeSample);
    };

    const handleLength = () => {
        if (!activeSample?.jobId) {
            addNotification('Najpierw wykonaj konwersję', 'error');
            return;
        }
        onRunLength?.(activeSample);
    };

    const handleGlomeruli = () => {
        if (!activeSample?.jobId) {
            addNotification('Najpierw wykonaj konwersję', 'error');
            return;
        }
        onRunGlomeruli?.(activeSample);
    };

    return (
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
                disabled={batchRunning || isConverting || hasAnyAnalysis}
                className={activeSample ? 'selected' : ''}
            >
                <img src="/folder-open.svg" width={20} height={20} alt="" />
                <span>Wybierz folder</span>
                {activeSample && <span className="checkmark">✓</span>}
            </button>

            {activeSample && (
                <div className="selected-folder">
                    <strong>Aktywna próbka:</strong> {activeSample.name}
                </div>
            )}

            <button
                disabled={batchRunning || isConverting || hasAnyAnalysis || !activeSample || activeSample.files.length === 0}
                id="convert"
                onClick={handleConvert}
                className={activeSample?.processStage === 'converted' ? 'completed' : ''}
            >
                <img src="/convert.svg" width={20} height={20} alt="" />
                <span>{isConverting ? 'Konwertowanie...' : 'Konwertuj'}</span>
                {activeSample?.processStage === 'converted' && <span className="checkmark">✓</span>}
            </button>

            <div className="analysis-section">
                <p>Analiza</p>

                <button
                    disabled={batchRunning || isConverting || notConverted || isFibrosisRunning}
                    id="fibrosis"
                    onClick={handleFibrosis}
                    className={activeSample?.fibrosisCompleted ? 'completed' : ''}
                >
                    <img src="/analyze.svg" width={20} height={20} alt="" />
                    <span>{isFibrosisRunning ? 'Analizowanie...' : 'Analizuj zwłóknienie'}</span>
                    {activeSample?.fibrosisCompleted && <span className="checkmark">✓</span>}
                </button>

                <button
                    disabled={batchRunning || isConverting || notConverted || isLengthRunning}
                    id="length"
                    onClick={handleLength}
                    className={activeSample?.lengthCompleted ? 'completed' : ''}
                >
                    <img src="/ruler.svg" width={20} height={20} alt="" />
                    <span>{isLengthRunning ? 'Analizowanie...' : 'Analizuj długość'}</span>
                    {activeSample?.lengthCompleted && <span className="checkmark">✓</span>}
                </button>

                <button
                    disabled={batchRunning || isConverting || notConverted || isGlomeruliRunning}
                    id="glomeruli"
                    onClick={handleGlomeruli}
                    className={activeSample?.glomeruliCompleted ? 'completed' : ''}
                >
                    <img src="/detect.svg" width={20} height={20} alt="" />
                    <span>{isGlomeruliRunning ? 'Wykrywanie...' : 'Wykryj kłębuszki'}</span>
                    {activeSample?.glomeruliCompleted && <span className="checkmark">✓</span>}
                </button>
            </div>
        </div>
    );
}
