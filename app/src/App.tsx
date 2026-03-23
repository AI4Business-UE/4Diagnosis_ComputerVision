import { useState } from 'react'
import './App.css'
import ControlPanel from './components/ControlPanel/ControlPanel'
import ResultsPanel from './components/ResultsPanel/ResultsPanel'
import ImageViewer from './components/ImageViewer/ImageViewer'
import { NotificationProvider } from './components/Notifications/NotificationContext'
import NotificationContainer from './components/Notifications/NotificationContainer'

interface ImageVersion {
  id: 'original' | 'fibrosis' | 'length'
  label: string
  url: string
}

function App() {
  const [imageVersions, setImageVersions] = useState<ImageVersion[]>([]);
  const [analysisResult, setAnalysisResult] = useState<any | null>(null);

  const handleTiffReady = (tiffUrl: string | null) => {
    if (!tiffUrl) {
      setImageVersions([])
      return
    }

    setImageVersions([
      { id: 'original', label: 'Oryginalny TIFF', url: tiffUrl }
    ])
  }

  const handleOverlayReady = (id: ImageVersion['id'], label: string, url: string) => {
    setImageVersions(prev => {
      const withoutCurrent = prev.filter(v => v.id !== id)
      return [...withoutCurrent, { id, label, url }]
    })
  }

  const handleAnalysisComplete = (data: any) => {
    setAnalysisResult((prev: any) => ({
      ...(prev ?? {}),
      ...(data ?? {}),
    }))
  }


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
          />

          <ImageViewer versions={imageVersions} />

          <ResultsPanel result={analysisResult} />
        </div>

        <NotificationContainer />
      </>
    </NotificationProvider>
  )
}

export default App
