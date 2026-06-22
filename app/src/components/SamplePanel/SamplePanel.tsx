import './SamplePanel.css';
import type { Sample } from '../../types/Sample';

interface SamplePanelProps {
  samples: Sample[];
  activeSampleId: string | null;
  onSelectSample: (sampleId: string) => void;
}

export default function SamplePanel({ samples, activeSampleId, onSelectSample }: SamplePanelProps) {
  return (
    <div className="sample-panel">
      <div className="sample-panel-header">
        <h3>Próbki</h3>
        <span className="sample-count">{samples.length}</span>
      </div>
      
      <div className="sample-list">
        {samples.length === 0 ? (
          <div className="no-samples">
            <p>Brak próbek</p>
            <p className="hint">Wybierz folder z próbkami</p>
          </div>
        ) : (
          samples.map((sample) => (
            <div
              key={sample.id}
              className={`sample-item ${activeSampleId === sample.id ? 'active' : ''} ${sample.processStage}`}
              onClick={() => onSelectSample(sample.id)}
            >
              <div className="sample-name">{sample.name}</div>
              <div className="sample-status">
                {sample.processStage === 'converted' && (
                  <span className="badge converted">✓</span>
                )}
                {sample.fibrosisCompleted && (
                  <span className="badge analysis">Z</span>
                )}
                {sample.lengthCompleted && (
                  <span className="badge analysis">D</span>
                )}
                {sample.glomerulesCompleted && (
                  <span className="badge analysis">K</span>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}