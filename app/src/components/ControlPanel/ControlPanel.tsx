import './ControlPanel.css'
import { useState } from 'react'

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

   const handleAnalyze = async () => {
  if (!jobId) {
    alert("Najpierw wykonaj konwersję");
    return;
  }

  const response = await fetch(`${IP}/api/analyze/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ job_id: jobId }),
  });

  const data = await response.json();
  console.log("ANALYSIS RESULT:", data);
    onAnalysisComplete?.(data);
};




    return (
        <div className="control-panel">
            <p>
                Panel sterowania
            </p>
            <input
                type="file"
                multiple
                // @ts-ignore
                webkitdirectory=""
                style={{ display: 'none' }}
                id="folder-input"
                onChange={handleFolderUpload}
            />
            

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

