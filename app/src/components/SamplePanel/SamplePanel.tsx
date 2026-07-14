import { useState } from 'react';
import './SamplePanel.css';
import { ChevronLeftIcon, ChevronRightIcon } from '../icons/Icons';
import type { Sample } from '../../types/Sample';

interface SamplePanelProps {
  samples: Sample[];
  activeSampleId: string | null;
  onSelectSample: (sampleId: string) => void;
  /** Sample currently being processed by the batch run, if any. */
  processingSampleId?: string | null;
}

export default function SamplePanel({
  samples,
  activeSampleId,
  onSelectSample,
  processingSampleId = null,
}: SamplePanelProps) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className={`sample-panel${collapsed ? ' collapsed' : ''}`}>
      <div className="sample-panel-header">
        {!collapsed && (
          <>
            <h3>Próbki</h3>
            <span className="sample-count">{samples.length}</span>
          </>
        )}
        <button
          className="collapse-toggle"
          onClick={() => setCollapsed(c => !c)}
          title={collapsed ? 'Pokaż listę próbek' : 'Ukryj listę próbek'}
          aria-label={collapsed ? 'Pokaż listę próbek' : 'Ukryj listę próbek'}
          aria-expanded={!collapsed}
        >
          {collapsed ? <ChevronRightIcon size={18} /> : <ChevronLeftIcon size={18} />}
        </button>
      </div>

      {collapsed ? (
        <div className="sample-rail">
          <span className="sample-count rail-count">{samples.length}</span>
          {samples.map((sample, index) => (
            <button
              key={sample.id}
              className={`rail-item${activeSampleId === sample.id ? ' active' : ''}${processingSampleId === sample.id ? ' processing' : ''}${sample.processStage === 'converted' ? ' converted' : ''}`}
              onClick={() => onSelectSample(sample.id)}
              title={sample.name}
              aria-label={sample.name}
            >
              {index + 1}
            </button>
          ))}
        </div>
      ) : (
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
                className={`sample-item ${activeSampleId === sample.id ? 'active' : ''} ${sample.processStage} ${processingSampleId === sample.id ? 'processing' : ''}`}
                onClick={() => onSelectSample(sample.id)}
              >
                <div className="sample-name">{sample.name}</div>
                <div className="sample-status">
                  {processingSampleId === sample.id && <span className="badge running">…</span>}
                  {sample.processStage === 'converted' && (
                    <span className="badge converted" title="Skonwertowana">✓</span>
                  )}
                  {sample.fibrosisCompleted && (
                    <span className="badge analysis" title="Zwłóknienie">Z</span>
                  )}
                  {sample.lengthCompleted && (
                    <span className="badge analysis" title="Długość">D</span>
                  )}
                  {sample.glomeruliCompleted && (
                    <span className="badge analysis" title="Kłębuszki">K</span>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
