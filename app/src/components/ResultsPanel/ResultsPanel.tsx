import './ResultsPanel.css'
import {
    FibrosisPieIcon,
    RulerIcon,
    GlomeruliCountIcon,
    GlomeruliHealthyIcon,
    GlomeruliScleroticIcon,
} from '../icons/Icons'
import type { ImageVersionId } from '../../types/Sample'

interface ResultsPanelProps {
    result: {
        length?: number;
        fibrosis_ratio?: number;
        /** Average across all slices — only present in all-slices mode */
        fibrosis_ratio_avg?: number;
        fibrosis_warning?: boolean;
        glomeruli_count?: number;
    } | null;
    glomeruliScanning?: boolean;
    glomeruliBreakdown?: { healthy: number; sclerotic: number };
    availableVersions?: ImageVersionId[];
    activeVersionId?: ImageVersionId | null;
    onSelectVersion?: (id: ImageVersionId) => void;
    width?: number;
}

export default function ResultsPanel({
    result,
    glomeruliScanning,
    glomeruliBreakdown,
    availableVersions = [],
    activeVersionId,
    onSelectVersion,
    width,
}: ResultsPanelProps) {
    const displayCount = glomeruliBreakdown != null
        ? glomeruliBreakdown.healthy + glomeruliBreakdown.sclerotic
        : result?.glomeruli_count;

    const hasAvg = result?.fibrosis_ratio_avg != null;

    const hasFibrosisView = availableVersions.includes('fibrosis');
    const hasLengthView = availableVersions.includes('length');
    const hasGlomeruliView = availableVersions.includes('glomeruli');

    const cardProps = (target: ImageVersionId, enabled: boolean) => {
        const isActive = enabled && activeVersionId === target;
        return {
            type: 'button' as const,
            disabled: !enabled,
            className: `result${enabled ? ' clickable' : ''}${isActive ? ' active-view' : ''}`,
            onClick: () => { if (enabled) onSelectVersion?.(target); },
            title: enabled ? 'Pokaż podgląd' : undefined,
        };
    };

    const panelStyle = width
        ? ({ '--results-panel-w': `${width}px` } as React.CSSProperties)
        : undefined;

    return (
        <div className="results-panel" style={panelStyle}>
            <p>Wyniki analiz</p>
            <div className="results-list">
                <button {...cardProps('fibrosis', hasFibrosisView)}>
                    <FibrosisPieIcon size={24} className="result-icon icon-fibrosis" />
                    <div className="result-info">
                        <h2>
                            Procent zwłóknienia
                            {result?.fibrosis_warning && (
                                <span className="fibrosis-warning-badge" title="Wyniki różnią się między skrawkami">⚠️</span>
                            )}
                        </h2>
                        <span id="zwloknienie" className="result-value">
                            {result?.fibrosis_ratio != null
                                ? `${(result.fibrosis_ratio * 100).toFixed(2)}%`
                                : "—"}
                        </span>
                        {hasAvg && (
                            <span className="fibrosis-avg">
                                Śr. wszystkich: {(result!.fibrosis_ratio_avg! * 100).toFixed(2)}%
                            </span>
                        )}
                    </div>
                </button>

                <button {...cardProps('length', hasLengthView)}>
                    <RulerIcon size={24} className="result-icon icon-length" />
                    <div className="result-info">
                        <h2>Długość tkanki</h2>
                        <span className="result-value">
                            {result?.length != null ? `${result.length.toFixed(3)} mm` : "—"}
                        </span>
                    </div>
                </button>

                <button {...cardProps('glomeruli', hasGlomeruliView)}>
                    <GlomeruliCountIcon size={24} className="result-icon icon-count" />
                    <div className="result-info">
                        <h2>Liczba kłębuszków</h2>
                        <span id="ilosc-klebuszkow" className="result-value">
                            {displayCount != null
                                ? <>{displayCount}{glomeruliScanning && <span className="scanning-indicator"> skanowanie...</span>}</>
                                : "—"}
                        </span>
                        {displayCount != null && glomeruliBreakdown != null && displayCount > 0 && (
                            <span className="glomeruli-split">
                                <span className="split-bar">
                                    <span
                                        className="split-healthy"
                                        style={{ width: `${(glomeruliBreakdown.healthy / displayCount) * 100}%` }}
                                    />
                                </span>
                                <span className="split-legend">
                                    {glomeruliBreakdown.healthy} niezwł. / {glomeruliBreakdown.sclerotic} zwł.
                                </span>
                            </span>
                        )}
                    </div>
                </button>

                <button {...cardProps('glomeruli', hasGlomeruliView)}>
                    <GlomeruliHealthyIcon size={24} className="result-icon icon-healthy" />
                    <div className="result-info">
                        <h2>Kłębuszki niezwłóknione</h2>
                        <span className="result-value breakdown-healthy">
                            {glomeruliBreakdown != null ? glomeruliBreakdown.healthy : "—"}
                        </span>
                    </div>
                </button>

                <button {...cardProps('glomeruli', hasGlomeruliView)}>
                    <GlomeruliScleroticIcon size={24} className="result-icon icon-sclerotic" />
                    <div className="result-info">
                        <h2>Kłębuszki zwłóknione</h2>
                        <span className="result-value breakdown-sclerotic">
                            {glomeruliBreakdown != null ? glomeruliBreakdown.sclerotic : "—"}
                        </span>
                    </div>
                </button>
            </div>
        </div>
    );
}
