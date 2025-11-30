import './ControlPanel.css'

export default function ControlPanel() {
    return (
        <div className="control-panel">
            <p>
                Panel sterowania
            </p>
            <button id='choose-folder'>
                <i className="fa-regular fa-folder-open"></i>
                <span>Wybierz folder</span>
            </button>
            <button id='convert'></button>
            <button id='analyse'></button>
            <button id='detect'></button>
        </div>
    );
}