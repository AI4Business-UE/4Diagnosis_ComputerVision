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

const IP = "http://127.0.0.1:8000";

interface ControlPanelProps {
    onDirectorySelect?: (directory: FileSystemDirectoryHandle) => void;
    onAnalysisComplete?: (data: any) => void;
}


export default function ControlPanel({ onAnalysisComplete }: ControlPanelProps) {
  const [folderStatus, setFolderStatus] = useState(false);
  const [selectedFolderName, setSelectedFolderName] = useState<string | null>(null);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const { addNotification, removeNotification } = useNotification();
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [processStage, setProcessStage] = useState<'initial' | 'folder_selected' | 'converted'>('initial');

    

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

        console.log('Wybrano plików:', files.length);
    };

    const handleConvert = async () => {
    if (uploadedFiles.length === 0) {
      alert('Wybierz folder');
      return;
    }

    setIsLoading(true);

    try {
      const formData = new FormData();

      for (const file of uploadedFiles) {
        const lower = file.name.toLowerCase();
        if (
          lower.endsWith('.mrxs') ||
          lower.endsWith('.dat') ||
          lower === 'slidedat.ini'
        ) {
          formData.append('files', file, file.name);
        }
      }

      const response = await fetch(`${IP}/api/convert/`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      console.log('Convert response:', data);
      if (data.job_id) {
        setJobId(data.job_id);
    }

      onAnalysisComplete?.(data);
    } catch (err) {
      console.error(err);
      alert('Błąd konwersji');
    } finally {
      setIsLoading(false);
    }
  }

    const handleMask = async () => {
        const loadingNotificationId = addNotification('Tworzenie maski...', 'loading');
        try {
            // TODO: Backend implementation for mask creation if needed
            setTimeout(() => {
                removeNotification(loadingNotificationId);
                addNotification('Maska utworzona!', 'success');
            }, 2000);
        } catch (err) {
            console.error('Błąd:', err);
            removeNotification(loadingNotificationId);
            addNotification('Błąd podczas tworzenia maski', 'error');
        }
    };

    const handleFibrosis = async () => {
        const loadingNotificationId = addNotification('Analizowanie włóknień...', 'loading');

        try {
            const result = await analyzeFibrosis();

            if (result.success) {
                removeNotification(loadingNotificationId);
                addNotification('Analiza włóknień zakończona!', 'success');
                console.log('Wyniki włóknień:', result.data);
            } else {
                throw new Error(result.error || 'Błąd analizy');
            }

        } catch (err) {
            console.error('Błąd analizy włóknień:', err);
            removeNotification(loadingNotificationId);
            addNotification('Błąd podczas analizy włóknień', 'error');
        }
    };

    const handleLength = async () => {
        const loadingNotificationId = addNotification('Analizowanie długości...', 'loading');

        try {
            const result = await analyzeLength();

            if (result.success) {
                removeNotification(loadingNotificationId);
                addNotification('Analiza długości zakończona!', 'success');
                console.log('Wyniki długości:', result.data);
            } else {
                throw new Error(result.error || 'Błąd analizy');
            }

        } catch (err) {
            console.error('Błąd analizy długości:', err);
            removeNotification(loadingNotificationId);
            addNotification('Błąd podczas analizy długości', 'error');
        }
    };

    const handleGlomerule = async () => {
        const loadingNotificationId = addNotification('Wykrywanie kłębuszków...', 'loading');

        try {
            const result = await detectGlomerules();

            if (result.success) {
                removeNotification(loadingNotificationId);
                addNotification('Wykrycie kłębuszków zakończone!', 'success');
                console.log('Wyniki kłębuszków:', result.data);
            } else {
                throw new Error(result.error || 'Błąd wykrywania');
            }

        } catch (err) {
            console.error('Błąd wykrywania kłębuszków:', err);
            removeNotification(loadingNotificationId);
            addNotification('Błąd podczas wykrywania kłębuszków', 'error');
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
                    webkitdirectory="true"
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

                    <button disabled={processStage !== 'converted'} id="fibrosis" onClick={handleFibrosis}>
                        <img
                            src={"/analyze.svg"}
                            width={20}
                            height={20}
                            alt=""
                            aria-hidden="true"
                            className="icon-analyze"
                        />
                        <span>Analizuj zwłóknienie</span>
                    </button>
                    <button disabled={processStage !== 'converted'} id="length" onClick={handleLength}>
                        <img
                            src={"/analyze.svg"}
                            width={20}
                            height={20}
                            alt=""
                            aria-hidden="true"
                            className="icon-analyze"
                        />
                        <span>Analizuj długość</span>
                    </button>
                    <button disabled={processStage !== 'converted'} id="glomerule" onClick={handleGlomerule}>
                        <img
                            src={"/detect.svg"}
                            width={20}
                            height={20}
                            alt=""
                            aria-hidden="true"
                            className="icon-detect"
                        />
                        <span>Wykryj kłębuszki</span>
                    </button>
                </div>
            </div>
        </>
    )
}

