import './LoadingScreen.css';

interface LoadingScreenProps {
  isVisible: boolean;
  progress?: number;
}

export default function LoadingScreen({ isVisible, progress = 0 }: LoadingScreenProps) {
  if (!isVisible) return null;

  return (
    <div className="loading-screen">
      <div className="loading-content">
        <div className="loading-spinner"></div>
        <h2>Konwertowanie plików...</h2>
        <p>Proszę czekać, ten proces może potrwać 1-5 minut</p>
        
        {progress > 0 && (
          <div className="progress-wrapper">
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progress}%` }}></div>
            </div>
            <div className="progress-text">{progress}%</div>
          </div>
        )}
      </div>
    </div>
  );
}
