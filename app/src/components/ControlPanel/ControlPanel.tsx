import './ControlPanel.css'
import { useState } from 'react'
import { useNotification } from '../Notifications/NotificationContext'
import LoadingScreen from '../LoadingScreen/LoadingScreen'
import {
  convertToTiff,
  analyzeFibrosis,
  analyzeLength,
  detectGlomerules,
} from '../../services/api'

const API_ORIGIN = 'http://127.0.0.1:8000';

interface ControlPanelProps {
    onTiffReady?: (tiffUrl: string | null) => void;
    onOverlayReady?: (id: 'fibrosis' | 'length' | 'glomeruli', label: string, url: string) => void;
    onAnalysisComplete?: (data: any) => void;
}

interface AnalysisResult {
  length?: number
  fibrosis_ratio?: number
  glomeruli_count?: number
}

export default function ControlPanel({ onAnalysisComplete, onTiffReady, onOverlayReady }: ControlPanelProps) {
  const [folderStatus, setFolderStatus] = useState(false);
  const [selectedFolderName, setSelectedFolderName] = useState<string | null>(null);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const { addNotification, removeNotification } = useNotification();
    const [loadingProgress] = useState(0);
  const [processStage, setProcessStage] = useState<'initial' | 'folder_selected' | 'converted'>('initial');
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult>({});
  const [fibrosisCompleted, setFibrosisCompleted] = useState(false);
  const [lengthCompleted, setLengthCompleted] = useState(false);
  const [glomerulesCompleted, setGlomerulesCompleted] = useState(false);

    const toResultImageUrl = (imagePath: string, jobId: string | null) => {
    if (!jobId) return null;

    const fileName = imagePath.split(/[/\\]/).pop();
    if (!fileName) return null;

    return `${API_ORIGIN}/api/result-image/${encodeURIComponent(jobId)}/${encodeURIComponent(fileName)}/`;
    };

    

    const handleSelectFolder = () => {
    const input = document.getElementById('folder-input') as HTMLInputElement | null;
    if (!input) return;

    input.value = ''; // pozwala wybrać ten sam folder ponownie
    input.click();
  };
    const handleFolderUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!e.target.files || e.target.files.length === 0) return;

        const files = Array.from(e.target.files);
        setUploadedFiles(files);
        setFolderStatus(true);

        const folderName = files[0].webkitRelativePath.split('/')[0];
        setSelectedFolderName(folderName);
        setProcessStage('folder_selected');

                addNotification(`Wybrano ${files.length} plików`, 'info', 2000);
    };

    const handleConvert = async () => {
    if (uploadedFiles.length === 0) {
            addNotification('Wybierz folder przed konwersją', 'error');
      return;
    }

    setIsLoading(true);
        const loadingId = addNotification('Konwersja do TIFF w toku...', 'loading');

    try {
            const result = await convertToTiff(uploadedFiles);

            if (!result.success || !result.data) {
                throw new Error(result.error || 'Nieznany błąd konwersji');
            }

            const data = result.data;
            if (data.job_id) {
                setJobId(data.job_id);
        setProcessStage('converted');
                // const previewUrl = data.mask_preview_url || data.tiff_url; // fallback to tiff if mask failed
                const previewUrl = data.mask_preview_url || data.tiff_url;
                const fullPreviewUrl = previewUrl ? `${API_ORIGIN}${previewUrl}` : null;
                onTiffReady?.(fullPreviewUrl);
                // onTiffReady?.(previewUrl);
//                 const tiffUrl = `${API_ORIGIN}${data.tiff_url}`;
//                 onTiffReady?.(tiffUrl);
                addNotification('Konwersja zakończona. TIFF załadowany.', 'success');
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

    
  }

     const handleFibrosis = async () => {

            if (!jobId) {
            addNotification('Najpierw wykonaj konwersję', 'error')
            return
            }

            const loadingId = addNotification('Analiza zwłóknienia...', 'loading')

            try {

            const result = await analyzeFibrosis(jobId)

            if (result.success && result.data) {

                setAnalysisResult(prev => ({
                ...prev,
                ...result.data
                }))

                onAnalysisComplete?.({
                ...analysisResult,
                ...result.data
                })

                if (typeof result.data.image_path === 'string' && result.data.image_path.length > 0) {
                    const overlayUrl = toResultImageUrl(result.data.image_path, jobId);
                    if (overlayUrl) {
                        onOverlayReady?.('fibrosis', 'Zwłóknienie (overlay)', overlayUrl);
                    }
                }


                setFibrosisCompleted(true);
                addNotification('Analiza zwłóknienia zakończona', 'success')
            } else {
                throw new Error(result.error || 'Błąd analizy zwłóknienia')
            }

            } catch (err) {
            const message = err instanceof Error ? err.message : 'Błąd analizy zwłóknienia'
            addNotification(message, 'error', 5000)

            } finally {

            removeNotification(loadingId)
            }
        }


    const handleLength = async () => {

        if (!jobId) {
            addNotification('Najpierw wykonaj konwersję', 'error');
            return;
        }

        const loadingNotificationId = addNotification('Analizowanie długości...', 'loading');

        try {
            const result = await analyzeLength(jobId);

            if (result.success && result.data) {
                setAnalysisResult(prev => ({
                    ...prev,
                    ...result.data
                }))

                onAnalysisComplete?.(result.data)

                if (typeof result.data.image_path === 'string' && result.data.image_path.length > 0) {
                    const overlayUrl = toResultImageUrl(result.data.image_path, jobId);
                    if (overlayUrl) {
                        onOverlayReady?.('length', 'Długość tkanki (overlay)', overlayUrl);
                    }
                }

                setLengthCompleted(true);
                removeNotification(loadingNotificationId);
                addNotification('Analiza długości zakończona!', 'success');
            } else {
                throw new Error(result.error || 'Błąd analizy długości');
            }

        } catch (err) {
            const message = err instanceof Error ? err.message : 'Błąd podczas analizy długości';
            removeNotification(loadingNotificationId);
            addNotification(message, 'error', 5000);
        }
    };

    const handleGlomerule = async () => {
        if (!jobId) {
            addNotification('Najpierw wykonaj konwersję', 'error');
            return;
        }

        const loadingNotificationId = addNotification('Wykrywanie kłębuszków...', 'loading');

        try {
            const result = await detectGlomerules(jobId);

            if (result.success && result.data) {
            const nextResult = {
                ...analysisResult,
                glomeruli_count: result.data.count ?? 0,
            };

            setAnalysisResult(nextResult);
            onAnalysisComplete?.(nextResult);

            if (typeof result.data.image_url === 'string' && result.data.image_url.length > 0) {
                const overlayUrl = `${API_ORIGIN}${result.data.image_url}`;
                onOverlayReady?.('glomeruli', `Kłębuszki (${result.data.count ?? 0})`, overlayUrl);
                }

            setGlomerulesCompleted(true);
            addNotification('Wykrycie kłębuszków zakończone!', 'success');
            } else {
                 throw new Error(result.error || 'Błąd wykrywania');
            }
            console.log('image_url from API:', result.data.image_url);
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Błąd podczas wykrywania kłębuszków';
            addNotification(message, 'error', 5000);
        } finally {
            removeNotification(loadingNotificationId);
        }
        };











    return (
        <>
            <LoadingScreen isVisible={isLoading} progress={loadingProgress} />
            
            <div className="control-panel">
                <p>
                    Panel sterowania
                </p>
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
                    aria-haspopup="true" 
                    aria-label="Wybierz folder"
                    onClick={handleSelectFolder}
                    className={folderStatus ? 'selected' : ''}
                >
                    <img
                        src={"/folder-open.svg"}
                        width={20}
                        height={20}
                        alt=""
                        aria-hidden="true"
                        className="icon-folder-open"
                    />
                    <span>Wybierz folder</span>
                    {folderStatus && <span className="checkmark">✓</span>}
                </button>
                       

                {selectedFolderName && (
                    <div className="selected-folder">
                        <strong>Wybrany folder:</strong> {selectedFolderName}
                    </div>
                )}

                <button 
                    disabled={processStage !== 'folder_selected'} 
                    id="convert"
                    onClick={handleConvert}
                    className={processStage !== 'folder_selected' && processStage !== 'initial' ? 'completed' : ''}
                >
                    <img
                        src={"/convert.svg"}
                        width={20}
                        height={20}
                        alt=""
                        aria-hidden="true"
                        className="icon-convert"
                    />
                    <span>Konwertuj</span>
                    {processStage !== 'folder_selected' && processStage !== 'initial' && (
                        <span className="checkmark">✓</span>
                    )}
                </button>

                <div className="analysis-section">
                    <p>Analiza</p>

                    <button disabled={processStage !== 'converted'} id="fibrosis" onClick={handleFibrosis} className={fibrosisCompleted ? 'completed' : ''}>
                        <img
                            src={"/analyze.svg"}
                            width={20}
                            height={20}
                            alt=""
                            aria-hidden="true"
                            className="icon-analyze"
                        />
                        <span>Analizuj zwłóknienie</span>
                        {fibrosisCompleted && <span className="checkmark">✓</span>}
                    </button>
                    <button disabled={processStage !== 'converted'} id="length" onClick={handleLength} className={lengthCompleted ? 'completed' : ''}>
                        <img
                            src={"/analyze.svg"}
                            width={20}
                            height={20}
                            alt=""
                            aria-hidden="true"
                            className="icon-analyze"
                        />
                        <span>Analizuj długość</span>
                        {lengthCompleted && <span className="checkmark">✓</span>}
                    </button>
                    <button disabled={processStage !== 'converted'} id="glomerule" onClick={handleGlomerule} className={glomerulesCompleted ? 'completed' : ''}>
                        <img
                            src={"/detect.svg"}
                            width={20}
                            height={20}
                            alt=""
                            aria-hidden="true"
                            className="icon-detect"
                        />
                        <span>Wykryj kłębuszki</span>
                        {glomerulesCompleted && <span className="checkmark">✓</span>}
                    </button>
                </div>
            </div>
        </>
    )
}


