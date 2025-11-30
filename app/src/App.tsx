import { useState } from 'react'
import './App.css'
import ControlPanel from './components/ControlPanel/ControlPanel'
import ResultsPanel from './components/ResultsPanel/ResultsPanel'
import ImageViewer from './components/ImageViewer/ImageViewer'

function App() {

  return (
    <>
      <div className="navbar">
        <img src="/4dfull.svg" alt="Logo" />
        <h1>ComputerVision</h1>
      </div>

      <div className="component-container">
        <ControlPanel />
        <ImageViewer />
        <ResultsPanel />
      </div>
    </>
  )
}

export default App
