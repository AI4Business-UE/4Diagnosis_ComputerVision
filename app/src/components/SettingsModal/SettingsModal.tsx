import { useEffect } from 'react';
import './SettingsModal.css';
import {
  DEFAULT_APP_SETTINGS,
  type AppSettings,
  type ConversionPreset,
} from '../../types/Settings';

interface SettingsModalProps {
  isOpen: boolean;
  settings: AppSettings;
  onChange: (next: AppSettings) => void;
  onClose: () => void;
}

const PRESET_OPTIONS: ConversionPreset[] = ['S', 'M', 'L', 'XL'];

export default function SettingsModal({
  isOpen,
  settings,
  onChange,
  onClose,
}: SettingsModalProps) {
  useEffect(() => {
    if (!isOpen) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      className="settings-backdrop"
      role="dialog"
      aria-modal="true"
      aria-labelledby="settings-modal-title"
      onClick={onClose}
    >
      <div className="settings-modal" onClick={e => e.stopPropagation()}>
        <div className="settings-header">
          <h2 id="settings-modal-title">Ustawienia</h2>
          <button
            type="button"
            className="settings-close-btn"
            onClick={onClose}
            aria-label="Zamknij"
          >
            ✕
          </button>
        </div>

        <div className="settings-body">
          <section className="settings-group">
            <div className="settings-group-header">
              <span className="settings-label">Jakość konwersji</span>
            </div>
            <div className="preset-grid" role="radiogroup" aria-label="Jakość konwersji">
              {PRESET_OPTIONS.map(preset => {
                const selected = settings.conversionPreset === preset;
                return (
                  <button
                    key={preset}
                    type="button"
                    role="radio"
                    aria-checked={selected}
                    className={`preset-card${selected ? ' active' : ''}`}
                    onClick={() => onChange({ ...settings, conversionPreset: preset })}
                  >
                    <span className="preset-size">{preset}</span>
                  </button>
                );
              })}
            </div>
          </section>

          <section className="settings-group">
            <label className="settings-toggle-row">
              <span className="settings-label">Grupowanie plastrów</span>
              <input
                type="checkbox"
                className="settings-switch"
                checked={settings.enableSliceGrouping}
                onChange={e => onChange({ ...settings, enableSliceGrouping: e.target.checked })}
              />
            </label>

            <label className="settings-toggle-row">
              <span className="settings-label">Uwzględnij kłębuszki w „Analizuj wszystko”</span>
              <input
                type="checkbox"
                className="settings-switch"
                checked={settings.includeGlomeruliInAnalyzeAll}
                onChange={e =>
                  onChange({ ...settings, includeGlomeruliInAnalyzeAll: e.target.checked })
                }
              />
            </label>
          </section>

          <section className="settings-group">
            <div className="settings-slider-block">
              <div className="settings-group-header">
                <span className="settings-label">Próg czułości zwłóknienia</span>
                <span className="settings-value">
                  {(settings.defaultFibrosisThreshold * 100).toFixed(1)}%
                </span>
              </div>
              <input
                type="range"
                className="settings-range"
                min={30}
                max={50}
                step={0.5}
                value={Number((settings.defaultFibrosisThreshold * 100).toFixed(1))}
                onChange={e =>
                  onChange({
                    ...settings,
                    defaultFibrosisThreshold: Number(e.target.value) / 100,
                  })
                }
              />
              <div className="settings-range-bounds">
                <span>30%</span>
                <span>40%</span>
                <span>50%</span>
              </div>
            </div>

            <div className="settings-slider-block">
              <div className="settings-group-header">
                <span className="settings-label">Próg ufności kłębuszków</span>
                <span className="settings-value">
                  {Math.round(settings.defaultGlomeruliConf * 100)}%
                </span>
              </div>
              <input
                type="range"
                className="settings-range"
                min={15}
                max={95}
                step={1}
                value={Math.round(settings.defaultGlomeruliConf * 100)}
                onChange={e =>
                  onChange({
                    ...settings,
                    defaultGlomeruliConf: Number(e.target.value) / 100,
                  })
                }
              />
              <div className="settings-range-bounds">
                <span>15%</span>
                <span>95%</span>
              </div>
            </div>
          </section>
        </div>

        <div className="settings-footer">
          <button
            type="button"
            className="settings-reset-btn"
            onClick={() => onChange({ ...DEFAULT_APP_SETTINGS })}
          >
            Przywróć domyślne
          </button>
          <button type="button" className="settings-done-btn" onClick={onClose}>
            Zapisz
          </button>
        </div>
      </div>
    </div>
  );
}
