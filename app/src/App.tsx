import { useState, useCallback, useEffect, useRef } from 'react'
import './App.css'
import ControlPanel from './components/ControlPanel/ControlPanel'
import ResultsPanel from './components/ResultsPanel/ResultsPanel'
import ImageViewer from './components/ImageViewer/ImageViewer'
import SamplePanel from './components/SamplePanel/SamplePanel'
import SettingsModal from './components/SettingsModal/SettingsModal'
import { NotificationProvider, useNotification } from './components/Notifications/NotificationContext'
import NotificationContainer from './components/Notifications/NotificationContainer'
import { BoltIcon, SettingsIcon } from './components/icons/Icons'
import {
  getActiveBaseImageUrl,
  isAbortError,
  runConvert,
  runFibrosis,
  runGlomeruli,
  runLength,
  runToggleGrouping,
} from './services/pipeline'
import type { SlideInfo, TileInfo } from './services/api'
import type { ImageVersionId, Sample } from './types/Sample'
import { loadAppSettings, saveAppSettings, type AppSettings } from './types/Settings'

type AnalysisKind = 'fibrosis' | 'length' | 'glomeruli';
const EMPTY_ANALYSIS_SET = new Set<AnalysisKind>();

const RESULTS_WIDTH_KEY = '4diagnosis_results_panel_w';
const DEFAULT_RESULTS_WIDTH = 264;
const MIN_RESULTS_WIDTH = 210;
const MAX_RESULTS_WIDTH = 440;

function loadResultsPanelWidth(): number {
  try {
    const raw = window.localStorage.getItem(RESULTS_WIDTH_KEY);
    if (!raw) return DEFAULT_RESULTS_WIDTH;
    const n = Number(raw);
    if (Number.isFinite(n)) {
      return Math.min(MAX_RESULTS_WIDTH, Math.max(MIN_RESULTS_WIDTH, n));
    }
  } catch {}
  return DEFAULT_RESULTS_WIDTH;
}

function AppContent() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [activeSampleId, setActiveSampleId] = useState<string | null>(null);
  const [appSettings, setAppSettings] = useState<AppSettings>(() => loadAppSettings());
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [activeVersionId, setActiveVersionId] = useState<ImageVersionId | null>(null);
  const [resultsPanelWidth, setResultsPanelWidth] = useState<number>(() => loadResultsPanelWidth());
  const [isResizingResults, setIsResizingResults] = useState(false);
  const resizeStartRef = useRef<{ x: number; startW: number } | null>(null);

  const activeSample = samples.find(s => s.id === activeSampleId) || null;
  const activeSampleIdRef = useRef<string | null>(activeSampleId);
  activeSampleIdRef.current = activeSampleId;

  const [confThresholds, setConfThresholds] = useState<{ 0: number; 1: number }>({
    0: appSettings.defaultGlomeruliConf,
    1: appSettings.defaultGlomeruliConf,
  });

  const [fibrosisRecalculating, setFibrosisRecalculating] = useState(false);
  const [groupingUpdating, setGroupingUpdating] = useState(false);
  const [convertingSampleIds, setConvertingSampleIds] = useState<Set<string>>(() => new Set());
  const [runningAnalyses, setRunningAnalyses] = useState<Record<string, Set<AnalysisKind>>>({});

  const [batchRunning, setBatchRunning] = useState(false);
  const [batchProgress, setBatchProgress] = useState<{ current: number; total: number } | null>(null);
  const [processingSampleId, setProcessingSampleId] = useState<string | null>(null);

  const { addNotification, updateNotification, removeNotification } = useNotification();

  const glomeruliList = activeSample?.glomeruli ?? [];
  const tilesScanned = activeSample?.glomeruliTiles ?? [];
  const slideInfo = activeSample?.glomeruliSlideInfo ?? null;
  const activeRunningAnalyses = activeSampleId ? (runningAnalyses[activeSampleId] ?? EMPTY_ANALYSIS_SET) : EMPTY_ANALYSIS_SET;
  const glomeruliScanning = activeRunningAnalyses.has('glomeruli');

  const setSampleAnalysisRunning = useCallback((sampleId: string, kind: AnalysisKind, running: boolean) => {
    setRunningAnalyses(prev => {
      const current = prev[sampleId];
      const nextSet = new Set(current ?? []);
      if (running) {
        nextSet.add(kind);
      } else {
        nextSet.delete(kind);
      }
      if (nextSet.size === 0) {
        if (!current) return prev;
        const { [sampleId]: _, ...rest } = prev;
        return rest;
      }
      return { ...prev, [sampleId]: nextSet };
    });
  }, []);

  const handleSettingsChange = useCallback((next: AppSettings) => {
    setAppSettings(next);
    saveAppSettings(next);
  }, []);

  useEffect(() => {
    if (!isResizingResults) return;
    const onMove = (e: MouseEvent) => {
      if (!resizeStartRef.current) return;
      const dx = resizeStartRef.current.x - e.clientX;
      const nextW = Math.min(
        MAX_RESULTS_WIDTH,
        Math.max(MIN_RESULTS_WIDTH, Math.round(resizeStartRef.current.startW + dx))
      );
      setResultsPanelWidth(nextW);
    };
    const onUp = () => {
      setIsResizingResults(false);
      resizeStartRef.current = null;
      try {
        window.localStorage.setItem(RESULTS_WIDTH_KEY, String(resultsPanelWidth));
      } catch {}
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [isResizingResults, resultsPanelWidth]);

  const handleStartResultsResize = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    resizeStartRef.current = { x: e.clientX, startW: resultsPanelWidth };
    setIsResizingResults(true);
  }, [resultsPanelWidth]);

  useEffect(() => {
    const s = samples.find(sample => sample.id === activeSampleId);
    setActiveVersionId(null);
    const minConf = s?.glomeruliSlideInfo?.conf ?? 0.15;
    const initConf = Math.max(minConf, appSettings.defaultGlomeruliConf);
    setConfThresholds({ 0: initConf, 1: initConf });
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
    setSamples(prev => prev.map(s => (s.id === sampleId ? { ...s, ...updates } : s)));
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

  const handleConvertSample = useCallback(async (sample: Sample) => {
    const targetId = sample.id;
    const settingsSnapshot = appSettings;
    const controller = new AbortController();

    setConvertingSampleIds(prev => {
      const next = new Set(prev);
      next.add(targetId);
      return next;
    });
    const loadingId = addNotification(
      `Konwersja: ${sample.name}...`,
      'loading',
      0,
      () => controller.abort(),
      targetId
    );

    try {
      const converted = await runConvert(
        sample.files,
        {
          conversionPreset: settingsSnapshot.conversionPreset,
          enableSliceGrouping: settingsSnapshot.enableSliceGrouping,
        },
        controller.signal
      );

      updateSample(targetId, {
        jobId: converted.jobId,
        processStage: 'converted',
        imageVersions: converted.imageVersions,
        slicesSummary: converted.slicesSummary,
        analysisResult: {},
        fibrosisCompleted: false,
        lengthCompleted: false,
        glomeruliCompleted: false,
        glomeruli: [],
        glomeruliTiles: [],
        glomeruliSlideInfo: undefined,
      });
      if (activeSampleIdRef.current === targetId) {
        setActiveVersionId(null);
      }

      addNotification(`Konwersja zakończona: ${sample.name}`, 'success');
    } catch (err) {
      if (isAbortError(err)) {
        addNotification(`Anulowano konwersję: ${sample.name}`, 'info', 3000);
      } else {
        const message = err instanceof Error ? err.message : 'Błąd konwersji';
        addNotification(`${sample.name}: ${message}`, 'error', 5000);
      }
    } finally {
      removeNotification(loadingId);
      setConvertingSampleIds(prev => {
        const next = new Set(prev);
        next.delete(targetId);
        return next;
      });
    }
  }, [appSettings, updateSample, addNotification, removeNotification]);

  const resetGlomeruliForSample = useCallback((sampleId: string) => {
    setSamples(prev => prev.map(s =>
      s.id !== sampleId ? s : {
        ...s,
        glomeruli: [],
        glomeruliTiles: [],
        glomeruliSlideInfo: undefined,
        imageVersions: s.imageVersions.filter(v => v.id !== 'glomeruli'),
      }
    ));
  }, []);

  const handleRunFibrosis = useCallback(async (sample: Sample) => {
    if (!sample.jobId) {
      addNotification('Najpierw wykonaj konwersję', 'error');
      return;
    }
    const targetId = sample.id;
    const jobId = sample.jobId;
    if (runningAnalyses[targetId]?.has('fibrosis')) return;

    const threshold = appSettings.defaultFibrosisThreshold;
    setSampleAnalysisRunning(targetId, 'fibrosis', true);
    const controller = new AbortController();
    const loadingId = addNotification(
      `Analiza zwłóknienia: ${sample.name}...`,
      'loading',
      0,
      () => controller.abort(),
      targetId
    );

    try {
      const { data, overlayUrl } = await runFibrosis(jobId, threshold, controller.signal);

      setSamples(prev => prev.map(s => {
        if (s.id !== targetId || s.jobId !== jobId) return s;
        return {
          ...s,
          fibrosisCompleted: true,
          analysisResult: {
            ...s.analysisResult,
            fibrosis_ratio: data.fibrosis_ratio,
            fibrosis_ratio_avg: data.fibrosis_ratio_avg,
            fibrosis_warning: data.fibrosis_warning ?? false,
            fibrosis_threshold: data.threshold,
          },
          imageVersions: overlayUrl
            ? [...s.imageVersions.filter(v => v.id !== 'fibrosis'), { id: 'fibrosis', label: 'Zwłóknienie', url: overlayUrl }]
            : s.imageVersions,
        };
      }));

      if (data.fibrosis_warning) {
        addNotification(
          `⚠️ Uwaga: wyniki zwłóknienia różnią się między slicami. Avg: ${data.fibrosis_ratio_avg != null ? (data.fibrosis_ratio_avg * 100).toFixed(1) : '?'}%, reprezentant: ${data.fibrosis_ratio != null ? (data.fibrosis_ratio * 100).toFixed(1) : '?'}%`,
          'error',
          8000
        );
      }

      addNotification(`Analiza zwłóknienia zakończona: ${sample.name}`, 'success');
    } catch (err) {
      if (isAbortError(err)) {
        addNotification(`Anulowano analizę zwłóknienia: ${sample.name}`, 'info', 3000);
      } else {
        const message = err instanceof Error ? err.message : 'Błąd analizy zwłóknienia';
        addNotification(`${sample.name}: ${message}`, 'error', 5000);
      }
    } finally {
      removeNotification(loadingId);
      setSampleAnalysisRunning(targetId, 'fibrosis', false);
    }
  }, [appSettings.defaultFibrosisThreshold, runningAnalyses, setSampleAnalysisRunning, addNotification, removeNotification]);

  const handleRunLength = useCallback(async (sample: Sample) => {
    if (!sample.jobId) {
      addNotification('Najpierw wykonaj konwersję', 'error');
      return;
    }
    const targetId = sample.id;
    const jobId = sample.jobId;
    if (runningAnalyses[targetId]?.has('length')) return;

    setSampleAnalysisRunning(targetId, 'length', true);
    const controller = new AbortController();
    const loadingId = addNotification(
      `Analizowanie długości: ${sample.name}...`,
      'loading',
      0,
      () => controller.abort(),
      targetId
    );

    try {
      const { data, overlayUrl } = await runLength(jobId, controller.signal);

      setSamples(prev => prev.map(s => {
        if (s.id !== targetId || s.jobId !== jobId) return s;
        return {
          ...s,
          lengthCompleted: true,
          analysisResult: {
            ...s.analysisResult,
            length: data.length,
          },
          imageVersions: overlayUrl
            ? [...s.imageVersions.filter(v => v.id !== 'length'), { id: 'length', label: 'Długość tkanki', url: overlayUrl }]
            : s.imageVersions,
        };
      }));

      addNotification(`Analiza długości zakończona: ${sample.name}`, 'success');
    } catch (err) {
      if (isAbortError(err)) {
        addNotification(`Anulowano analizę długości: ${sample.name}`, 'info', 3000);
      } else {
        const message = err instanceof Error ? err.message : 'Błąd podczas analizy długości';
        addNotification(`${sample.name}: ${message}`, 'error', 5000);
      }
    } finally {
      removeNotification(loadingId);
      setSampleAnalysisRunning(targetId, 'length', false);
    }
  }, [runningAnalyses, setSampleAnalysisRunning, addNotification, removeNotification]);

  const handleRunGlomeruli = useCallback(async (sample: Sample) => {
    if (!sample.jobId) {
      addNotification('Najpierw wykonaj konwersję', 'error');
      return;
    }
    const targetId = sample.id;
    const jobId = sample.jobId;
    if (runningAnalyses[targetId]?.has('glomeruli')) return;

    const defaultConf = appSettings.defaultGlomeruliConf;
    resetGlomeruliForSample(targetId);
    setSampleAnalysisRunning(targetId, 'glomeruli', true);
    const controller = new AbortController();
    const loadingId = addNotification(
      `Wykrywanie kłębuszków: ${sample.name}...`,
      'loading',
      0,
      () => controller.abort(),
      targetId
    );

    try {
      const result = await runGlomeruli(
        jobId,
        {
          onSlideInfo: (info: SlideInfo) => {
            if (activeSampleIdRef.current === targetId) {
              const initConf = Math.max(info.conf, defaultConf);
              setConfThresholds({ 0: initConf, 1: initConf });
            }
            setSamples(prev => prev.map(s => {
              if (s.id !== targetId || s.jobId !== jobId) return s;
              const baseUrl = getActiveBaseImageUrl(s.imageVersions);
              return {
                ...s,
                glomeruliSlideInfo: info,
                imageVersions: baseUrl
                  ? [...s.imageVersions.filter(v => v.id !== 'glomeruli'), { id: 'glomeruli', label: 'Kłębuszki', url: baseUrl }]
                  : s.imageVersions,
              };
            }));
          },
          onGlomeruli: (batch) => {
            setSamples(prev => prev.map(s =>
              s.id !== targetId || s.jobId !== jobId
                ? s
                : { ...s, glomeruli: [...(s.glomeruli ?? []), ...batch] }
            ));
          },
          onTiles: (tiles: TileInfo[]) => {
            setSamples(prev => prev.map(s =>
              s.id !== targetId || s.jobId !== jobId
                ? s
                : { ...s, glomeruliTiles: [...(s.glomeruliTiles ?? []), ...tiles] }
            ));
          },
        },
        controller.signal
      );

      setSamples(prev => prev.map(s =>
        s.id !== targetId || s.jobId !== jobId
          ? s
          : {
              ...s,
              glomeruli: result.glomeruli,
              glomeruliCompleted: true,
              analysisResult: { ...s.analysisResult, glomeruli_count: result.count },
            }
      ));

      addNotification(`Wykryto ${result.count} kłębuszków: ${sample.name}`, 'success');
    } catch (err) {
      if (isAbortError(err)) {
        addNotification(`Anulowano wykrywanie kłębuszków: ${sample.name}`, 'info', 3000);
      } else {
        const message = err instanceof Error ? err.message : 'Błąd wykrywania kłębuszków';
        addNotification(`${sample.name}: ${message}`, 'error', 5000);
      }
    } finally {
      removeNotification(loadingId);
      setSampleAnalysisRunning(targetId, 'glomeruli', false);
    }
  }, [appSettings.defaultGlomeruliConf, runningAnalyses, resetGlomeruliForSample, setSampleAnalysisRunning, addNotification, removeNotification]);

  const handleToggleGrouping = useCallback(async (grouped: boolean) => {
    if (!activeSampleId || !activeSample?.jobId || groupingUpdating || batchRunning) return;
    if (runningAnalyses[activeSampleId]?.size) return;

    setGroupingUpdating(true);
    try {
      const existingRawUrl = activeSample.imageVersions.find(v => v.id === 'raw')?.url ?? null;
      const { imageVersions, slicesSummary } = await runToggleGrouping(
        activeSample.jobId,
        grouped,
        existingRawUrl
      );

      updateSample(activeSampleId, {
        imageVersions,
        slicesSummary,
        analysisResult: {},
        fibrosisCompleted: false,
        lengthCompleted: false,
        glomeruliCompleted: false,
        glomeruli: [],
        glomeruliTiles: [],
        glomeruliSlideInfo: undefined,
      });

      addNotification(
        grouped ? 'Przywrócono grupowanie' : 'Wyłączono grupowanie',
        'info',
        3500
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Błąd zmiany grupowania';
      addNotification(message, 'error', 5000);
    } finally {
      setGroupingUpdating(false);
    }
  }, [activeSampleId, activeSample, groupingUpdating, batchRunning, runningAnalyses, updateSample, addNotification]);

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
      if (overlayUrl) addOverlay(activeSampleId, 'fibrosis', 'Zwłóknienie', overlayUrl);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Błąd przeliczania zwłóknienia';
      addNotification(message, 'error', 5000);
    } finally {
      setFibrosisRecalculating(false);
    }
  }, [activeSampleId, activeSample?.jobId, mergeAnalysis, addOverlay, addNotification]);

  /** Full pipeline for one sample; steps already done are skipped so a re-run resumes. */
  const processSample = useCallback(async (
    sample: Sample,
    currentSettings: AppSettings,
    signal?: AbortSignal
  ) => {
    const id = sample.id;
    let jobId = sample.jobId;
    let activeBaseUrl = getActiveBaseImageUrl(sample.imageVersions);

    if (!jobId || sample.processStage !== 'converted') {
      const converted = await runConvert(
        sample.files,
        {
          conversionPreset: currentSettings.conversionPreset,
          enableSliceGrouping: currentSettings.enableSliceGrouping,
        },
        signal
      );
      jobId = converted.jobId;
      activeBaseUrl = converted.activeBaseUrl;
      updateSample(id, {
        jobId,
        processStage: 'converted',
        imageVersions: converted.imageVersions,
        slicesSummary: converted.slicesSummary,
      });
    }

    if (signal?.aborted) throw new DOMException('Anulowano', 'AbortError');

    if (!sample.fibrosisCompleted) {
      const { data, overlayUrl } = await runFibrosis(
        jobId,
        currentSettings.defaultFibrosisThreshold,
        signal
      );
      mergeAnalysis(id, {
        fibrosis_ratio: data.fibrosis_ratio,
        fibrosis_ratio_avg: data.fibrosis_ratio_avg,
        fibrosis_warning: data.fibrosis_warning ?? false,
        fibrosis_threshold: data.threshold,
      });
      if (overlayUrl) addOverlay(id, 'fibrosis', 'Zwłóknienie', overlayUrl);
      updateSample(id, { fibrosisCompleted: true });
    }

    if (signal?.aborted) throw new DOMException('Anulowano', 'AbortError');

    if (!sample.lengthCompleted) {
      const { data, overlayUrl } = await runLength(jobId, signal);
      mergeAnalysis(id, { length: data.length });
      if (overlayUrl) addOverlay(id, 'length', 'Długość tkanki', overlayUrl);
      updateSample(id, { lengthCompleted: true });
    }

    if (signal?.aborted) throw new DOMException('Anulowano', 'AbortError');

    if (currentSettings.includeGlomeruliInAnalyzeAll && !sample.glomeruliCompleted) {
      resetGlomeruliForSample(id);
      setSampleAnalysisRunning(id, 'glomeruli', true);

      try {
        const result = await runGlomeruli(
          jobId,
          {
            onSlideInfo: (info) => {
              if (activeSampleIdRef.current === id) {
                const initConf = Math.max(info.conf, currentSettings.defaultGlomeruliConf);
                setConfThresholds({ 0: initConf, 1: initConf });
              }
              updateSample(id, { glomeruliSlideInfo: info });
              if (activeBaseUrl) addOverlay(id, 'glomeruli', 'Kłębuszki', activeBaseUrl);
            },
            onGlomeruli: (batch) => {
              setSamples(prev => prev.map(s =>
                s.id !== id ? s : { ...s, glomeruli: [...(s.glomeruli ?? []), ...batch] }
              ));
            },
            onTiles: (tiles) => {
              setSamples(prev => prev.map(s =>
                s.id !== id ? s : { ...s, glomeruliTiles: [...(s.glomeruliTiles ?? []), ...tiles] }
              ));
            },
          },
          signal
        );

        updateSample(id, {
          glomeruli: result.glomeruli,
          glomeruliCompleted: true,
        });
        mergeAnalysis(id, { glomeruli_count: result.count });
      } finally {
        setSampleAnalysisRunning(id, 'glomeruli', false);
      }
    }
  }, [updateSample, mergeAnalysis, addOverlay, resetGlomeruliForSample, setSampleAnalysisRunning]);

  /** Converts and analyses every sample, one after another. */
  const handleAnalyzeAll = useCallback(async () => {
    if (batchRunning || samples.length === 0) return;

    const queue = samples;
    const settingsSnapshot = appSettings;
    const controller = new AbortController();
    setBatchRunning(true);
    const loadingId = addNotification(
      `Analiza wszystkich próbek (0/${queue.length})...`,
      'loading',
      0,
      () => controller.abort()
    );
    let failed = 0;
    let cancelled = false;

    try {
      for (let i = 0; i < queue.length; i++) {
        if (controller.signal.aborted) {
          cancelled = true;
          break;
        }
        const sample = queue[i];
        setBatchProgress({ current: i + 1, total: queue.length });
        updateNotification(loadingId, `Analiza wszystkich próbek (${i + 1}/${queue.length}): ${sample.name}...`);
        setProcessingSampleId(sample.id);
        setActiveSampleId(sample.id);

        try {
          await processSample(sample, settingsSnapshot, controller.signal);
        } catch (err) {
          if (isAbortError(err)) {
            cancelled = true;
            break;
          }
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

    if (cancelled) {
      addNotification('Anulowano analizę wszystkich próbek', 'info', 4000);
    } else if (failed === 0) {
      addNotification(`Przeanalizowano ${queue.length} ${queue.length === 1 ? 'próbkę' : 'próbek'}`, 'success');
    } else {
      addNotification(`Zakończono z błędami: ${failed} z ${queue.length} próbek`, 'error', 6000);
    }
  }, [batchRunning, samples, appSettings, processSample, addNotification, updateNotification, removeNotification]);

  const glomeruliBreakdown = glomeruliList.length > 0 ? (() => {
    const filtered = glomeruliList.filter(g => g.conf >= confThresholds[g.cls as 0 | 1]);
    return {
      healthy: filtered.filter(g => g.cls === 0).length,
      sclerotic: filtered.filter(g => g.cls === 1).length,
    };
  })() : undefined;

  const busySampleIds = new Set<string>(convertingSampleIds);
  for (const [sampleId, kinds] of Object.entries(runningAnalyses)) {
    if (kinds.size > 0) busySampleIds.add(sampleId);
  }

  return (
    <>
      <div className="navbar">
        <img src="/logo.svg" alt="Logo" />
        <h1>ComputerVision</h1>

        <div className="navbar-actions">
          <button
            type="button"
            className="settings-button"
            onClick={() => setSettingsOpen(true)}
            title="Ustawienia"
          >
            <SettingsIcon size={18} />
            <span>Ustawienia</span>
          </button>

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

          <button
            type="button"
            className="new-analysis-button"
            onClick={() => window.location.reload()}
            title="Resetuje wszystkie próbki i rozpoczyna nową analizę"
            aria-label="Rozpocznij nową analizę"
          >
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
              <path d="M15 9C15 12.3137 12.3137 15 9 15C5.68629 15 3 12.3137 3 9C3 5.68629 5.68629 3 9 3C10.8364 3 12.4768 3.82312 13.5962 5.125M13.5 3V5.5H11" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <span>Nowa analiza</span>
          </button>
        </div>
      </div>

      <div className="component-container">
        <SamplePanel
          samples={samples}
          activeSampleId={activeSampleId}
          onSelectSample={setActiveSampleId}
          processingSampleId={processingSampleId}
          busySampleIds={busySampleIds}
        />

        <ControlPanel
          key={`control-${activeSampleId ?? 'empty'}`}
          activeSample={activeSample}
          batchRunning={batchRunning}
          isConverting={Boolean(activeSample && convertingSampleIds.has(activeSample.id))}
          runningAnalyses={activeRunningAnalyses}
          onSamplesDetected={handleSamplesDetected}
          onConvertSample={handleConvertSample}
          onRunFibrosis={handleRunFibrosis}
          onRunLength={handleRunLength}
          onRunGlomeruli={handleRunGlomeruli}
        />

        <ImageViewer
          key={`viewer-${activeSampleId ?? 'empty'}`}
          versions={activeSample?.imageVersions || []}
          activeVersionId={activeVersionId}
          onActiveVersionChange={setActiveVersionId}
          hasActiveSample={Boolean(activeSample)}
          isConverting={Boolean(activeSample && convertingSampleIds.has(activeSample.id))}
          slicesSummary={activeSample?.slicesSummary ?? null}
          groupingUpdating={groupingUpdating || activeRunningAnalyses.size > 0}
          onToggleGrouping={handleToggleGrouping}
          glomeruli={glomeruliList}
          slideInfo={slideInfo}
          tilesScanned={tilesScanned}
          confThresholds={confThresholds}
          onConfThresholdsChange={setConfThresholds}
          fibrosisRatio={activeSample?.analysisResult?.fibrosis_ratio}
          fibrosisThreshold={
            activeSample?.analysisResult?.fibrosis_threshold ?? appSettings.defaultFibrosisThreshold
          }
          fibrosisRecalculating={fibrosisRecalculating || activeRunningAnalyses.has('fibrosis')}
          onFibrosisThresholdCommit={handleFibrosisThresholdCommit}
        />

        <div
          className={`results-resize-handle${isResizingResults ? ' dragging' : ''}`}
          onMouseDown={handleStartResultsResize}
          onDoubleClick={() => {
            setResultsPanelWidth(DEFAULT_RESULTS_WIDTH);
            try { window.localStorage.setItem(RESULTS_WIDTH_KEY, String(DEFAULT_RESULTS_WIDTH)); } catch {}
          }}
          title="Zmień szerokość panelu"
          role="separator"
          aria-orientation="vertical"
          aria-label="Zmień szerokość panelu wyników"
        />

        <ResultsPanel
          result={activeSample?.analysisResult || null}
          glomeruliScanning={glomeruliScanning}
          glomeruliBreakdown={glomeruliBreakdown}
          availableVersions={(activeSample?.imageVersions || []).map(v => v.id)}
          activeVersionId={activeVersionId}
          onSelectVersion={setActiveVersionId}
          width={resultsPanelWidth}
        />
      </div>

      <SettingsModal
        isOpen={settingsOpen}
        settings={appSettings}
        onChange={handleSettingsChange}
        onClose={() => setSettingsOpen(false)}
      />

      <NotificationContainer activeSampleId={activeSampleId} />
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
