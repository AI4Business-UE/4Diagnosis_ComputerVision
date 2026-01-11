import './ControlPanel.css'
import { useState } from 'react'

const IP = "http://127.0.0.1:8000";

interface WindowWithDirectoryPicker extends Window {
    showDirectoryPicker?: () => Promise<FileSystemDirectoryHandle>;
}

interface ControlPanelProps {
    onDirectorySelect?: (directory: FileSystemDirectoryHandle) => void;
    onAnalysisComplete?: (data: any) => void;
}


export default function ControlPanel({ onDirectorySelect, onAnalysisComplete}: ControlPanelProps) {
    const [folderStatus, setFolderStatus] = useState<boolean>(false);
    const [selectedFolderName, setSelectedFolderName] = useState<string | null>(null);
    const [directory, setDirectory] = useState<FileSystemDirectoryHandle | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(false)
    
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
    }
    const handleFolderUpload = async (
  e: React.ChangeEvent<HTMLInputElement>
) => {
  if (!e.target.files) return;

  const formData = new FormData();

  for (const file of Array.from(e.target.files)) {
    const lower = file.name.toLowerCase();
    if (
      lower.endsWith('.mrxs') ||
      lower.endsWith('.dat') ||
      lower === 'slidedat.ini'
    ) {
      formData.append('files', file, file.name);
      console.log('UPLOAD:', file.webkitRelativePath);
    }
  }

  const response = await fetch(`${IP}/api/convert/`, {
    method: 'POST',
    body: formData
  });

  console.log(await response.json());
};



    
async function scanFolderRecursively(
  dirHandle: FileSystemDirectoryHandle,
  formData: FormData
) {
  console.log('SCAN DIR:', dirHandle.name);

  // @ts-ignore
  for await (const [name, entry] of dirHandle.entries()) {
    if (entry.kind === 'file') {
      const lower = name.toLowerCase();

      if (lower.endsWith('.mrxs') || lower.endsWith('.dat')) {
        const file = await entry.getFile();
        formData.append('files', file, name);
        console.log('Dodano do wysyłki:', name);
      }
    }

    if (entry.kind === 'directory') {
      await scanFolderRecursively(entry, formData);
    }
  }

  // 🔴 TYLKO TU jest sens szukać ini
  try {
    const iniHandle = await dirHandle.getFileHandle('Slidedat.ini');
    const iniFile = await iniHandle.getFile();
    formData.append('files', iniFile, 'Slidedat.ini');
    console.log('Dodano do wysyłki: Slidedat.ini');
  } catch (e) {
    console.warn('Brak Slidedat.ini w:', dirHandle.name);
  }
}






        const handleConvert = async () => {
        if (!directory) {
            alert('Wybierz folder');
            return;
        }

        const formData = new FormData();
        await scanFolderRecursively(directory, formData);

        const response = await fetch(`${IP}/api/convert/`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        console.log(data);
        };



    const handleAnalyze = async () => {
        if (!directory) {
            alert('Nie wybrano folderu');
            return;
        }
        setIsLoading(true);

        try {
            const formData = new FormData();
            const dirHandle = directory as FileSystemDirectoryHandle;
            let hasSVS = false; 

            // @ts-ignore 
            for await (const entry of dirHandle.values()) {
                if (entry.kind === 'file' && entry.name.endsWith('.mrxs')) {
                    const file = await entry.getFile();
                    console.log("INI FOUND:", file.name);
                    formData.append('files', file);
                }
             }
            console.log('FormData entries:');
            for (const [key, value] of formData.entries()) {
                console.log(key, value);
            }

            // const response = await fetch(IP, {  // BEZ /api/convert/ na końcu!
            //     method: "POST", 
            //     body: formData 
            //  });

            // if (!response.ok) throw new Error("Błąd serwera");
            // const data = await response.json();
            // onAnalysisComplete?.(data);

        } catch (error) {
            console.error('Błąd podczas analizy:", error');
            alert('Nie udało się przeprowadzić analizy.');
        } finally {
            setIsLoading(false);
        }
    };




    return (
        <div className="control-panel">
            <p>
                Panel sterowania
            </p>

            <button 
                id="choose-folder" 
                  // @ts-ignore
                 webkitdirectory=""
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

            <button disabled={!folderStatus} id="convert" onClick={handleConvert}>
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
            <button disabled={!folderStatus} id="analyze" onClick={handleAnalyze}>
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

