import { useState, useCallback } from 'react'
import './App.css'
import ControlPanel from './components/ControlPanel/ControlPanel'
import ResultsPanel from './components/ResultsPanel/ResultsPanel'
import ImageViewer from './components/ImageViewer/ImageViewer'
import SamplePanel from './components/SamplePanel/SamplePanel'
import { NotificationProvider } from './components/Notifications/NotificationContext'
import NotificationContainer from './components/Notifications/NotificationContainer'
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
    if (newSamples.length > 0) {
      setActiveSampleId(newSamples[0].id);
    }
  }, []);

  const updateSample = useCallback((sampleId: string, updates: Partial<Sample>) => {
    setSamples(prev => prev.map(sample => 
      sample.id === sampleId 
        ? { ...sample, ...updates }
        : sample
    ));
  }, []);

  const handleTiffReady = useCallback((tiffUrl: string | null) => {
    if (!activeSampleId) return;

    if (!tiffUrl) {
      updateSample(activeSampleId, { imageVersions: [] });
      return;
    }

    updateSample(activeSampleId, {
      imageVersions: [{ id: 'original', label: 'Oryginalny TIFF', url: tiffUrl }]
    });
  }, [activeSampleId, updateSample]);

  const handleOverlayReady = useCallback((
    id: ImageVersion['id'], 
    label: string, 
    url: string
  ) => {
    if (!activeSampleId) return;

    setSamples(prev => prev.map(sample => {
      if (sample.id !== activeSampleId) return sample;

      const withoutCurrent = sample.imageVersions.filter(v => v.id !== id);
      return {
        ...sample,
        imageVersions: [...withoutCurrent, { id, label, url }]
      };
    }));
  }, [activeSampleId]);

  const handleAnalysisComplete = useCallback((data: any) => {
    console.log('=== handleAnalysisComplete ===');
    console.log('Nowe dane:', data);
    
    if (!activeSampleId) {
        console.log('Brak activeSampleId');
        return;
    }

    setSamples(prev => {
        return prev.map(sample => {
            if (sample.id !== activeSampleId) {
                return sample;
            }

            const merged = {
                ...sample.analysisResult,
                ...data,
            };

            console.log('ID próbki:', sample.id);
            console.log('Stare wyniki:', sample.analysisResult);
            console.log('Nowe dane:', data);
            console.log('Scalone:', merged);

            return {
                ...sample,
                analysisResult: merged
            };
        });
    });
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
    
    updateSample(activeSampleId, {
      [`${type}Completed`]: completed
    } as Partial<Sample>);
  }, [activeSampleId, updateSample]);

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
          />

          <ImageViewer versions={activeSample?.imageVersions || []} />

          <ResultsPanel result={activeSample?.analysisResult || null} />
        </div>

        <NotificationContainer />
      </>
    </NotificationProvider>
  )
}

export default App