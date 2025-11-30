import { useState } from 'react'
import './App.css'
import ControlPanel from './components/ControlPanel/ControlPanel'

function App() {

  return (
    <>
      <div className="navbar">
        <img src="/logo.svg" alt="Logo" />
        <h1>ComputerVision</h1>
      </div>

      <ControlPanel />
    </>
  )
}

export default App
