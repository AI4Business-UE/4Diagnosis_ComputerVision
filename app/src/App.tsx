import { useState } from 'react'
import './App.css'
import ControlPanel from './components/ControlPanel/ControlPanel'
import ResultsPanel from './components/ResultsPanel/ResultsPanel'
import ImageViewer from './components/ImageViewer/ImageViewer'
import { NotificationProvider } from './components/Notifications/NotificationContext'
import NotificationContainer from './components/Notifications/NotificationContainer'

function App() {
  const [directory, setDirectory] = useState<FileSystemDirectoryHandle | null>(null);

  return (
    <NotificationProvider>
      <>
        <div className="navbar">
          <img src="/4dfull.svg" alt="Logo" />
          <h1>ComputerVision</h1>
        </div>

        <div className="component-container">
          <ControlPanel onDirectorySelect={setDirectory} />
          <ImageViewer directory={directory} />
          <ResultsPanel />
        </div>

        <NotificationContainer />
      </>
    </NotificationProvider>
  )
}

export default App
