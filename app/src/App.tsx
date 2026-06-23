import { useState, useCallback, useEffect } from 'react'
import './App.css'
import ControlPanel from './components/ControlPanel/ControlPanel'
import ResultsPanel from './components/ResultsPanel/ResultsPanel'
import ImageViewer from './components/ImageViewer/ImageViewer'
import SamplePanel from './components/SamplePanel/SamplePanel'
import { NotificationProvider } from './components/Notifications/NotificationContext'
import NotificationContainer from './components/Notifications/NotificationContainer'
import type { Glomerulus, SlideInfo, TileInfo } from './services/api'
import type { Sample } from './types/Sample'

interface ImageVersion {
  id: 'original' | 'fibrosis' | 'length' | 'glomeruli'
  label: string
  url: string
}

function App() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [activeSampleId, setActiveSampleId] = useState<string | null>(null);

  const activeSample = samples.find(s => s.id === activeSampleId) || null;

  const [glomeruliScanning, setGlomeruliScanning] = useState(false);
  const [glomeruliList, setGlomeruliList] = useState<Glomerulus[]>([]);
  const [slideInfo, setSlideInfo] = useState<SlideInfo | null>(null);
  const [tilesScanned, setTilesScanned] = useState<TileInfo[]>([]);
  const [confThresholds, setConfThresholds] = useState<{ 0: number; 1: number }>({ 0: 0.15, 1: 0.15 });

  useEffect(() => {
    const s = samples.find(s => s.id === activeSampleId);
    setGlomeruliList(s?.glomeruli ?? []);
    setTilesScanned(s?.glomeruliTiles ?? []);
    setSlideInfo(s?.glomeruliSlideInfo ?? null);
    setConfThresholds(s?.glomeruliSlideInfo ? { 0: s.glomeruliSlideInfo.conf, 1: s.glomeruliSlideInfo.conf } : { 0: 0.15, 1: 0.15 });
    setGlomeruliScanning(false);
  }, [activeSampleId]);

  const handleSamplesDetected = useCallback((detectedSamples: Array<{ name: string; files: File[] }>) => {
    const newSamples: Sample[] = detectedSamples.map((sample, index) => ({
      id: `sample-${Date.now()}-${index}`,
      name: sample.name,
      folderPath: sample.files[0]?.webkitRelativePath.split('/').slice(0, -1).join('/') || '',
      files: sample.files,
      jobId: null,
      processStage: 'folder_selected',
      analysisResult: {},
      fibrosisCompleted: false,
      lengthCompleted: false,
      glomerulesCompleted: false,
      imageVersions: [],
    }));
    setSamples(newSamples);
    if (newSamples.length > 0) setActiveSampleId(newSamples[0].id);
  }, []);

  const updateSample = useCallback((sampleId: string, updates: Partial<Sample>) => {
    setSamples(prev => prev.map(s => s.id === sampleId ? { ...s, ...updates } : s));
  }, []);

  const handleTiffReady = useCallback((tiffUrl: string | null) => {
    if (!activeSampleId) return;
    if (!tiffUrl) { updateSample(activeSampleId, { imageVersions: [] }); return; }
    updateSample(activeSampleId, {
      imageVersions: [{ id: 'original', label: 'Oryginalny TIFF', url: tiffUrl }]
    });
  }, [activeSampleId, updateSample]);

  const handleOverlayReady = useCallback((id: ImageVersion['id'], label: string, url: string) => {
    if (!activeSampleId) return;
    setSamples(prev => prev.map(s => {
      if (s.id !== activeSampleId) return s;
      return { ...s, imageVersions: [...s.imageVersions.filter(v => v.id !== id), { id, label, url }] };
    }));
  }, [activeSampleId]);

  const handleAnalysisComplete = useCallback((data: any) => {
    if (!activeSampleId) return;
    setSamples(prev => prev.map(s =>
      s.id !== activeSampleId ? s : { ...s, analysisResult: { ...s.analysisResult, ...data } }
    ));
  }, [activeSampleId]);

  const handleStageChange = useCallback((stage: Sample['processStage']) => {
    if (!activeSampleId) return;
    updateSample(activeSampleId, { processStage: stage });
  }, [activeSampleId, updateSample]);

  const handleJobIdChange = useCallback((jobId: string) => {
    if (!activeSampleId) return;
    updateSample(activeSampleId, { jobId });
  }, [activeSampleId, updateSample]);

  const handleAnalysisStatusChange = useCallback((type: 'fibrosis' | 'length' | 'glomerules', completed: boolean) => {
    if (!activeSampleId) return;
    updateSample(activeSampleId, { [`${type}Completed`]: completed } as Partial<Sample>);
  }, [activeSampleId, updateSample]);

  const glomeruliBreakdown = glomeruliList.length > 0 ? (() => {
    const filtered = glomeruliList.filter(g => g.conf >= confThresholds[g.cls as 0 | 1]);
    return {
      healthy: filtered.filter(g => g.cls === 0).length,
      sclerotic: filtered.filter(g => g.cls === 1).length,
    };
  })() : undefined;

  return (
    <NotificationProvider>
      <>
        <div className="navbar">
          <img src="/logo.svg" alt="Logo" />
          <h1>ComputerVision</h1>
        </div>

        <div className="component-container">
          <SamplePanel
            samples={samples}
            activeSampleId={activeSampleId}
            onSelectSample={setActiveSampleId}
          />

          <ControlPanel
            key={activeSampleId}
            activeSample={activeSample}
            onSamplesDetected={handleSamplesDetected}
            onTiffReady={handleTiffReady}
            onOverlayReady={handleOverlayReady}
            onAnalysisComplete={handleAnalysisComplete}
            onStageChange={handleStageChange}
            onJobIdChange={handleJobIdChange}
            onAnalysisStatusChange={handleAnalysisStatusChange}
            onGlomeruliScanning={setGlomeruliScanning}
            onGlomeruliDetected={(batch) => setGlomeruliList(prev => [...prev, ...batch])}
            onSlideInfo={(info) => {
              setSlideInfo(info);
              setConfThresholds({ 0: info.conf, 1: info.conf });
              if (activeSampleId) {
                updateSample(activeSampleId, { glomeruliSlideInfo: info });
                const originalUrl = activeSample?.imageVersions.find(v => v.id === 'original')?.url;
                if (originalUrl) handleOverlayReady('glomeruli', 'Kłębuszki (overlay)', originalUrl);
              }
            }}
            onGlomeruliReset={() => {
              setGlomeruliList([]);
              setTilesScanned([]);
              if (activeSampleId) {
                setSamples(prev => prev.map(s =>
                  s.id !== activeSampleId ? s : {
                    ...s,
                    glomeruli: [],
                    glomeruliTiles: [],
                    glomeruliSlideInfo: undefined,
                    imageVersions: s.imageVersions.filter(v => v.id !== 'glomeruli'),
                  }
                ));
              }
            }}
            onTilesUpdate={(tiles: TileInfo[]) => {
              setTilesScanned(prev => {
                const updated = [...prev, ...tiles];
                if (activeSampleId) updateSample(activeSampleId, { glomeruliTiles: updated });
                return updated;
              });
            }}
            onFinalGlomeruliList={(list) => {
              setGlomeruliList(list);
              if (activeSampleId) updateSample(activeSampleId, { glomeruli: list });
            }}
          />

          <ImageViewer
            versions={activeSample?.imageVersions || []}
            glomeruli={glomeruliList}
            slideInfo={slideInfo}
            tilesScanned={tilesScanned}
            confThresholds={confThresholds}
            onConfThresholdsChange={setConfThresholds}
          />

          <ResultsPanel
            result={activeSample?.analysisResult || null}
            glomeruliScanning={glomeruliScanning}
            glomeruliBreakdown={glomeruliBreakdown}
          />
        </div>

        <NotificationContainer />
      </>
    </NotificationProvider>
  )
}

export default App
