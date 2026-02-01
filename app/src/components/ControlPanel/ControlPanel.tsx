import './ControlPanel.css'
import { useState } from 'react'
import { useNotification } from '../Notifications/NotificationContext'
import LoadingScreen from '../LoadingScreen/LoadingScreen'

interface WindowWithDirectoryPicker extends Window {
    showDirectoryPicker?: () => Promise<FileSystemDirectoryHandle>;
}

interface ControlPanelProps {
    onDirectorySelect?: (directory: FileSystemDirectoryHandle) => void;
}

export default function ControlPanel({ onDirectorySelect }: ControlPanelProps) {

    type ProcessStage = 'initial' | 'folder_selected' | 'converted' | 'mask_created';

    const [folderStatus, setFolderStatus] = useState<boolean>(false);
    const [selectedFolderName, setSelectedFolderName] = useState<string | null>(null);
    const [directory, setDirectory] = useState<FileSystemDirectoryHandle | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [loadingProgress, setLoadingProgress] = useState(0);
    const [processStage, setProcessStage] = useState<ProcessStage>('initial');
    const { addNotification, removeNotification } = useNotification();
    
    const handleSelectFolder = async () => {
        const win = window as WindowWithDirectoryPicker;

        if (!win.showDirectoryPicker) {
            addNotification('Przeglądarka nie obsługuje showDirectoryPicker', 'error');
            return;
        }

        try {
            const getDirectory = await win.showDirectoryPicker();
            
            setDirectory(getDirectory);
            setSelectedFolderName(getDirectory.name);
            setFolderStatus(true);
            setProcessStage('folder_selected');
            
            if (onDirectorySelect) {
                onDirectorySelect(getDirectory);
            }

            addNotification(`Folder wybrany: ${getDirectory.name}`, 'success');

        } catch (err) {
            if (err instanceof Error && err.name === 'AbortError') {
                console.log('Wybór folderu anulowany.');
            } else {
                console.error('Błąd:', err);
                addNotification('Błąd podczas wyboru folderu', 'error');
            }

            setFolderStatus(false);
            setSelectedFolderName(null);
            setDirectory(null);
            setProcessStage('initial');
        }
    };

    const handleConvert = async () => {
        if (!directory) {
            addNotification('Najpierw wybierz folder', 'error');
            return;
        }

        setIsLoading(true);
        setLoadingProgress(0);
        const loadingNotificationId = addNotification('Konwertowanie plików na format TIFF...', 'loading');

        try {
            // TODO: Wysłanie zapytania do backendu
            // const response = await fetch('/api/convert', {
            //     method: 'POST',
            //     headers: { 'Content-Type': 'application/json' },
            //     body: JSON.stringify({ directory: directory.name })
            // });

            // Symulacja postępu dla demonstracji
            const progressInterval = setInterval(() => {
                setLoadingProgress(prev => {
                    if (prev >= 95) {
                        clearInterval(progressInterval);
                        return prev;
                    }
                    return prev + Math.random() * 15;
                });
            }, 500);

            // Symulacja końcowego czasu konwersji
            setTimeout(() => {
                clearInterval(progressInterval);
                setLoadingProgress(100);
                setIsLoading(false);
                setProcessStage('converted');
                removeNotification(loadingNotificationId);
                addNotification('Konwertowanie zakończone!', 'success');
            }, 3000);

        } catch (err) {
            console.error('Błąd konwersji:', err);
            removeNotification(loadingNotificationId);
            addNotification('Błąd podczas konwertowania', 'error');
            setIsLoading(false);
        }
    };

    const handleMask = async () => {
        const loadingNotificationId = addNotification('Tworzenie maski...', 'loading');
        try {
            // TODO: Wysłanie zapytania do backendu
            setTimeout(() => {
                removeNotification(loadingNotificationId);
                setProcessStage('mask_created');
                addNotification('Maska utworzona!', 'success');
            }, 2000);
        } catch (err) {
            console.error('Błąd:', err);
            removeNotification(loadingNotificationId);
            addNotification('Błąd podczas tworzenia maski', 'error');
        }
    };

    return (
        <>
            <LoadingScreen isVisible={isLoading} progress={loadingProgress} />
            
            <div className="control-panel">
                <p>
                    Panel sterowania
                </p>

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
                <button 
                    disabled={processStage !== 'converted'} 
                    id="mask"
                    onClick={handleMask}
                    className={processStage === 'mask_created' ? 'completed' : ''}
                >
                    <img
                        src={"/mask.svg"}
                        width={20}
                        height={20}
                        alt=""
                        aria-hidden="true"
                        className="icon-mask"
                    />
                    <span>Stwórz maskę</span>
                    {processStage === 'mask_created' && (
                        <span className="checkmark">✓</span>
                    )}
                </button>
                <button disabled={processStage !== 'mask_created'} id="analyze">
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
                <button disabled={processStage !== 'mask_created'} id="analyze">
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
                <button disabled={processStage !== 'mask_created'} id="detect">
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
        </>
    )
}