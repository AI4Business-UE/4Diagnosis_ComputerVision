import { useState, useCallback, useEffect } from 'react'
import './App.css'
import ControlPanel from './components/ControlPanel/ControlPanel'
import ResultsPanel from './components/ResultsPanel/ResultsPanel'
import ImageViewer from './components/ImageViewer/ImageViewer'
import SamplePanel from './components/SamplePanel/SamplePanel'
import { NotificationProvider, useNotification } from './components/Notifications/NotificationContext'
import NotificationContainer from './components/Notifications/NotificationContainer'
import { BoltIcon } from './components/icons/Icons'
import { runConvert, runFibrosis, runLength, runGlomeruli } from './services/pipeline'
import type { Glomeruli, SlideInfo, TileInfo } from './services/api'
import type { Sample } from './types/Sample'

type ImageVersionId = Sample['imageVersions'][number]['id'];

function AppContent() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [activeSampleId, setActiveSampleId] = useState<string | null>(null);

  const activeSample = samples.find(s => s.id === activeSampleId) || null;

  const [glomeruliScanning, setGlomeruliScanning] = useState(false);
  const [glomeruliList, setGlomeruliList] = useState<Glomeruli[]>([]);
  const [slideInfo, setSlideInfo] = useState<SlideInfo | null>(null);
  const [tilesScanned, setTilesScanned] = useState<TileInfo[]>([]);
  const [confThresholds, setConfThresholds] = useState<{ 0: number; 1: number }>({ 0: 0.15, 1: 0.15 });

  const [fibrosisRecalculating, setFibrosisRecalculating] = useState(false);

  const [batchRunning, setBatchRunning] = useState(false);
  const [batchProgress, setBatchProgress] = useState<{ current: number; total: number } | null>(null);
  const [processingSampleId, setProcessingSampleId] = useState<string | null>(null);

  const { addNotification, removeNotification } = useNotification();

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
      glomeruliCompleted: false,
      imageVersions: [],
    }));
    setSamples(newSamples);
    if (newSamples.length > 0) setActiveSampleId(newSamples[0].id);
  }, []);

  const updateSample = useCallback((sampleId: string, updates: Partial<Sample>) => {
    setSamples(prev => prev.map(s => s.id === sampleId ? { ...s, ...updates } : s));
  }, []);

  const mergeAnalysis = useCallback((sampleId: string, data: Sample['analysisResult']) => {
    setSamples(prev => prev.map(s =>
      s.id !== sampleId ? s : { ...s, analysisResult: { ...s.analysisResult, ...data } }
    ));
  }, []);

  const addOverlay = useCallback((sampleId: string, id: ImageVersionId, label: string, url: string) => {
    setSamples(prev => prev.map(s =>
      s.id !== sampleId ? s : { ...s, imageVersions: [...s.imageVersions.filter(v => v.id !== id), { id, label, url }] }
    ));
  }, []);

  const handleTiffReady = useCallback((tiffUrl: string | null) => {
    if (!activeSampleId) return;
    updateSample(activeSampleId, {
      imageVersions: tiffUrl ? [{ id: 'original', label: 'Oryginalny TIFF', url: tiffUrl }] : []
    });
  }, [activeSampleId, updateSample]);

  const handleOverlayReady = useCallback((id: ImageVersionId, label: string, url: string) => {
    if (!activeSampleId) return;
    addOverlay(activeSampleId, id, label, url);
  }, [activeSampleId, addOverlay]);

  const handleAnalysisComplete = useCallback((data: Sample['analysisResult']) => {
    if (!activeSampleId) return;
    mergeAnalysis(activeSampleId, data);
  }, [activeSampleId, mergeAnalysis]);

  const handleStageChange = useCallback((stage: Sample['processStage']) => {
    if (!activeSampleId) return;
    updateSample(activeSampleId, { processStage: stage });
  }, [activeSampleId, updateSample]);

  const handleJobIdChange = useCallback((jobId: string) => {
    if (!activeSampleId) return;
    updateSample(activeSampleId, { jobId });
  }, [activeSampleId, updateSample]);

  const handleAnalysisStatusChange = useCallback((type: 'fibrosis' | 'length' | 'glomeruli', completed: boolean) => {
    if (!activeSampleId) return;
    updateSample(activeSampleId, { [`${type}Completed`]: completed } as Partial<Sample>);
  }, [activeSampleId, updateSample]);

  /** Re-runs fibrosis analysis with a user-adjusted threshold (dragged on the results bar). */
  const handleFibrosisThresholdCommit = useCallback(async (threshold: number) => {
    if (!activeSampleId || !activeSample?.jobId) return;

    setFibrosisRecalculating(true);
    try {
      const { data, overlayUrl } = await runFibrosis(activeSample.jobId, threshold);
      mergeAnalysis(activeSampleId, {
        fibrosis_ratio: data.fibrosis_ratio,
        fibrosis_ratio_avg: data.fibrosis_ratio_avg,
        fibrosis_warning: data.fibrosis_warning ?? false,
        fibrosis_threshold: data.threshold ?? threshold,
      });
      if (overlayUrl) addOverlay(activeSampleId, 'fibrosis', 'Zwłóknienie (overlay)', overlayUrl);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Błąd przeliczania zwłóknienia';
      addNotification(message, 'error', 5000);
    } finally {
      setFibrosisRecalculating(false);
    }
  }, [activeSampleId, activeSample?.jobId, mergeAnalysis, addOverlay, addNotification]);

  const resetGlomeruliForSample = useCallback((sampleId: string) => {
    setGlomeruliList([]);
    setTilesScanned([]);
    setSamples(prev => prev.map(s =>
      s.id !== sampleId ? s : {
        ...s,
        glomeruli: [],
        glomeruliTiles: [],
        glomeruliSlideInfo: undefined,
        imageVersions: s.imageVersions.filter(v => v.id !== 'glomeruli' && v.id !== 'glom_grid'),
      }
    ));
  }, []);

  /** Full pipeline for one sample; steps already done are skipped so a re-run resumes. */
  const processSample = useCallback(async (sample: Sample) => {
    const id = sample.id;
    let jobId = sample.jobId;
    let originalUrl = sample.imageVersions.find(v => v.id === 'original')?.url ?? null;

    if (!jobId || sample.processStage !== 'converted') {
      const converted = await runConvert(sample.files);
      jobId = converted.jobId;
      originalUrl = converted.previewUrl;
      updateSample(id, {
        jobId,
        processStage: 'converted',
        imageVersions: originalUrl
          ? [{ id: 'original', label: 'Oryginalny TIFF', url: originalUrl }]
          : [],
      });
    }

    if (!sample.fibrosisCompleted) {
      const { data, overlayUrl } = await runFibrosis(jobId);
      mergeAnalysis(id, {
        fibrosis_ratio: data.fibrosis_ratio,
        fibrosis_ratio_avg: data.fibrosis_ratio_avg,
        fibrosis_warning: data.fibrosis_warning ?? false,
        fibrosis_threshold: data.threshold,
      });
      if (overlayUrl) addOverlay(id, 'fibrosis', 'Zwłóknienie (overlay)', overlayUrl);
      updateSample(id, { fibrosisCompleted: true });
    }

    if (!sample.lengthCompleted) {
      const { data, overlayUrl } = await runLength(jobId);
      mergeAnalysis(id, { length: data.length });
      if (overlayUrl) addOverlay(id, 'length', 'Długość tkanki (overlay)', overlayUrl);
      updateSample(id, { lengthCompleted: true });
    }

    if (!sample.glomeruliCompleted) {
      resetGlomeruliForSample(id);
      setGlomeruliScanning(true);

      const collectedTiles: TileInfo[] = [];
      try {
        const result = await runGlomeruli(jobId, {
          onSlideInfo: (info) => {
            setSlideInfo(info);
            setConfThresholds({ 0: info.conf, 1: info.conf });
            updateSample(id, { glomeruliSlideInfo: info });
            if (originalUrl) addOverlay(id, 'glomeruli', 'Kłębuszki (overlay)', originalUrl);
          },
          onGlomeruli: (batch) => setGlomeruliList(prev => [...prev, ...batch]),
          onTiles: (tiles) => {
            collectedTiles.push(...tiles);
            setTilesScanned(prev => [...prev, ...tiles]);
          },
        });

        setGlomeruliList(result.glomeruli);
        updateSample(id, {
          glomeruli: result.glomeruli,
          glomeruliTiles: collectedTiles,
          glomeruliCompleted: true,
        });
        mergeAnalysis(id, { glomeruli_count: result.count });
        if (result.gridUrl) addOverlay(id, 'glom_grid', 'Porównanie kłębuszków (siatka)', result.gridUrl);
      } finally {
        setGlomeruliScanning(false);
      }
    }
  }, [updateSample, mergeAnalysis, addOverlay, resetGlomeruliForSample]);

  /** Converts and analyses every sample, one after another. */
  const handleAnalyzeAll = useCallback(async () => {
    if (batchRunning || samples.length === 0) return;

    const queue = samples;
    setBatchRunning(true);
    const loadingId = addNotification(`Analiza wszystkich próbek (0/${queue.length})...`, 'loading');
    let failed = 0;

    try {
      for (let i = 0; i < queue.length; i++) {
        const sample = queue[i];
        setBatchProgress({ current: i + 1, total: queue.length });
        setProcessingSampleId(sample.id);
        setActiveSampleId(sample.id);

        try {
          await processSample(sample);
        } catch (err) {
          failed++;
          const message = err instanceof Error ? err.message : 'Nieznany błąd';
          addNotification(`${sample.name}: ${message}`, 'error', 6000);
        }
      }
    } finally {
      removeNotification(loadingId);
      setProcessingSampleId(null);
      setBatchProgress(null);
      setBatchRunning(false);
    }

    if (failed === 0) {
      addNotification(`Przeanalizowano ${queue.length} ${queue.length === 1 ? 'próbkę' : 'próbek'}`, 'success');
    } else {
      addNotification(`Zakończono z błędami: ${failed} z ${queue.length} próbek`, 'error', 6000);
    }
  }, [batchRunning, samples, processSample, addNotification, removeNotification]);

  const glomeruliBreakdown = glomeruliList.length > 0 ? (() => {
    const filtered = glomeruliList.filter(g => g.conf >= confThresholds[g.cls as 0 | 1]);
    return {
      healthy: filtered.filter(g => g.cls === 0).length,
      sclerotic: filtered.filter(g => g.cls === 1).length,
    };
  })() : undefined;

  return (
    <>
      <div className="navbar">
        <img src="/logo.svg" alt="Logo" />
        <h1>ComputerVision</h1>

        <button
          className={`analyze-all-button${batchRunning ? ' running' : ''}`}
          onClick={handleAnalyzeAll}
          disabled={samples.length === 0 || batchRunning}
          title="Konwertuje i analizuje każdą próbkę po kolei"
        >
          <BoltIcon size={18} />
          <span>
            {batchRunning && batchProgress
              ? `Analiza… ${batchProgress.current}/${batchProgress.total}`
              : 'Analizuj wszystko'}
          </span>
        </button>
      </div>

      <div className="component-container">
        <SamplePanel
          samples={samples}
          activeSampleId={activeSampleId}
          onSelectSample={setActiveSampleId}
          processingSampleId={processingSampleId}
        />

        <ControlPanel
          key={activeSampleId}
          activeSample={activeSample}
          batchRunning={batchRunning}
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
            if (activeSampleId) resetGlomeruliForSample(activeSampleId);
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
          fibrosisRatio={activeSample?.analysisResult?.fibrosis_ratio}
          fibrosisThreshold={activeSample?.analysisResult?.fibrosis_threshold}
          fibrosisRecalculating={fibrosisRecalculating}
          onFibrosisThresholdCommit={handleFibrosisThresholdCommit}
        />

        <ResultsPanel
          result={activeSample?.analysisResult || null}
          glomeruliScanning={glomeruliScanning}
          glomeruliBreakdown={glomeruliBreakdown}
        />
      </div>

      <NotificationContainer />
    </>
  );
}

export default function App() {
  return (
    <NotificationProvider>
      <AppContent />
    </NotificationProvider>
  );
}
