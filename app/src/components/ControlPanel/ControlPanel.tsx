import './ControlPanel.css'
import { useState } from 'react'

interface WindowWithDirectoryPicker extends Window {
    showDirectoryPicker?: () => Promise<FileSystemDirectoryHandle>;
}

interface ControlPanelProps {
    onDirectorySelect?: (directory: FileSystemDirectoryHandle) => void;
}

export default function ControlPanel({ onDirectorySelect }: ControlPanelProps) {

    const [folderStatus, setFolderStatus] = useState<boolean>(false);
    const [selectedFolderName, setSelectedFolderName] = useState<string | null>(null);
    const [directory, setDirectory] = useState<FileSystemDirectoryHandle | null>(null);
    
    const handleSelectFolder = async () => {
        const win = window as WindowWithDirectoryPicker;

        if (!win.showDirectoryPicker) {
            alert('Przeglądarka nie obsługuje showDirectoryPicker');
            return;
        }

        try {
            const getDirectory = await win.showDirectoryPicker();
            
            setDirectory(getDirectory);
            setSelectedFolderName(getDirectory.name);
            setFolderStatus(true);
            
            if (onDirectorySelect) {
                onDirectorySelect(getDirectory);
            }

            alert(`Wybrany folder: ${getDirectory.name}`);

        } catch (err) {
            if (err instanceof Error && err.name === 'AbortError') {
                console.log('Wybór folderu anulowany.');
            } else {
                console.error('Błąd:', err);
                alert('Wystąpił błąd podczas próby wyboru folderu.');
            }

            setFolderStatus(false);
            setSelectedFolderName(null);
            setDirectory(null);
        }
    };

    return (
        <div className="control-panel">
            <p>
                Panel sterowania
            </p>

            <button 
                id="choose-folder" 
                aria-haspopup="true" 
                aria-label="Wybierz folder"
                onClick={handleSelectFolder}
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
            </button>

            {selectedFolderName && (
                <div className="selected-folder">
                    <strong>Wybrany folder:</strong> {selectedFolderName}
                </div>
            )}

            <button disabled={!folderStatus} id="convert">
                <img
                    src={"/convert.svg"}
                    width={20}
                    height={20}
                    alt=""
                    aria-hidden="true"
                    className="icon-convert"
                />
                <span>Konwertuj</span>
            </button>
            <button disabled={!folderStatus} id="analyze">
                <img
                    src={"/analyze.svg"}
                    width={20}
                    height={20}
                    alt=""
                    aria-hidden="true"
                    className="icon-analyze"
                />
                <span>Analizuj tkankę</span>
            </button>
            <button disabled={!folderStatus} id="detect">
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
    )
}