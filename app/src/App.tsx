import { useState } from 'react'
import './App.css'
import ControlPanel from './components/ControlPanel/ControlPanel'
import ResultsPanel from './components/ResultsPanel/ResultsPanel'
import ImageViewer from './components/ImageViewer/ImageViewer'
import { NotificationProvider } from './components/Notifications/NotificationContext'
import NotificationContainer from './components/Notifications/NotificationContainer'
import type { Glomerulus, SlideInfo, TileInfo } from './services/api'

interface ImageVersion {
  id: 'original' | 'fibrosis' | 'length' | 'glomeruli'
  label: string
  url: string
}

function App() {
  const [imageVersions, setImageVersions] = useState<ImageVersion[]>([]);
  const [analysisResult, setAnalysisResult] = useState<any | null>(null);
  const [glomeruliScanning, setGlomeruliScanning] = useState(false);
  const [glomeruliList, setGlomeruliList] = useState<Glomerulus[]>([]);
  const [slideInfo, setSlideInfo] = useState<SlideInfo | null>(null);
  const [tilesScanned, setTilesScanned] = useState<TileInfo[]>([]);
  const [confThresholds, setConfThresholds] = useState<{ 0: number; 1: number }>({ 0: 0.15, 1: 0.15 });

  const handleTiffReady = (tiffUrl: string | null) => {
    if (!tiffUrl) {
      setImageVersions([]);
      return;
    }

    setImageVersions([
      { id: 'original', label: 'Oryginalny TIFF', url: tiffUrl }
    ]);
  };

  const handleOverlayReady = (id: ImageVersion['id'], label: string, url: string) => {
    setImageVersions(prev => {
      const withoutCurrent = prev.filter(v => v.id !== id);
      return [...withoutCurrent, { id, label, url }];
    });
    console.log('overlay url raw:', url);
  };

  const handleAnalysisComplete = (data: any) => {
    setAnalysisResult((prev: any) => ({
      ...(prev ?? {}),
      ...(data ?? {}),
    }));
  };

  return (
    <NotificationProvider>
      <>
        <div className="navbar">
          <img src="/logo.svg" alt="Logo" />
          <h1>ComputerVision</h1>
        </div>

        <div className="component-container">
          <ControlPanel
            onTiffReady={handleTiffReady}
            onOverlayReady={handleOverlayReady}
            onAnalysisComplete={handleAnalysisComplete}
            onGlomeruliScanning={setGlomeruliScanning}
            onGlomeruliDetected={(batch) => setGlomeruliList(prev => [...prev, ...batch])}
            onSlideInfo={(info) => {
              setSlideInfo(info);
              // Inicjuj progi ufności z parametru conf modelu — min suwaka = conf.
              // Jeśli zmienisz conf w GlomeruliProcessor, tutaj też się zaktualizuje automatycznie.
              setConfThresholds({ 0: info.conf, 1: info.conf });
            }}
            onGlomeruliReset={() => { setGlomeruliList([]); setTilesScanned([]); }}
            onTilesUpdate={(tiles: TileInfo[]) => setTilesScanned(prev => [...prev, ...tiles])}
            onFinalGlomeruliList={(list) => setGlomeruliList(list)}
          />

          <ImageViewer
            versions={imageVersions}
            glomeruli={glomeruliList}
            slideInfo={slideInfo}
            tilesScanned={tilesScanned}
            confThresholds={confThresholds}
            onConfThresholdsChange={setConfThresholds}
          />

          <ResultsPanel
            result={analysisResult}
            glomeruliScanning={glomeruliScanning}
            glomeruliBreakdown={glomeruliList.length > 0 ? (() => {
              const filtered = glomeruliList.filter(g => g.conf >= confThresholds[g.cls as 0 | 1]);
              return {
                healthy: filtered.filter(g => g.cls === 0).length,
                sclerotic: filtered.filter(g => g.cls === 1).length,
              };
            })() : undefined}
          />
        </div>

        <NotificationContainer />
      </>
    </NotificationProvider>
  )
}

export default App