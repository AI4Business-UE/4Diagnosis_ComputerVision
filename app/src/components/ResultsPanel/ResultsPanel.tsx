import './ResultsPanel.css'
import {
    FibrosisPieIcon,
    RulerIcon,
    GlomeruliCountIcon,
    GlomeruliHealthyIcon,
    GlomeruliScleroticIcon,
} from '../icons/Icons'

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
}


const handleReset = () => { window.location.reload(); };

export default function ResultsPanel({ result, glomeruliScanning, glomeruliBreakdown }: ResultsPanelProps) {
    const displayCount = glomeruliBreakdown != null
        ? glomeruliBreakdown.healthy + glomeruliBreakdown.sclerotic
        : result?.glomeruli_count;

    const hasAvg = result?.fibrosis_ratio_avg != null;

    return (
        <div className="results-panel">
            <p>Wyniki analiz</p>
            <div className="results-list">
            <div className="result">
                <FibrosisPieIcon className="result-icon icon-fibrosis" />
                <div className="result-info">
                    <h2>
                        Procent zwłóknienia
                        {result?.fibrosis_warning && (
                            <span className="fibrosis-warning-badge" title="Wyniki różnią się między slicami">⚠️</span>
                        )}
                    </h2>
                    <span id="zwloknienie">
                        {result?.fibrosis_ratio != null
                            ? `${(result.fibrosis_ratio * 100).toFixed(2)}%`
                            : "—"}
                    </span>
                    {hasAvg && (
                        <span className="fibrosis-avg">
                            Śr. wszystkich sliców: {(result!.fibrosis_ratio_avg! * 100).toFixed(2)}%
                        </span>
                    )}
                </div>
            </div>
            <div className="result">
                <RulerIcon className="result-icon icon-length" />
                <div className="result-info">
                    <h2>Długość tkanki</h2>
                    <span>
                        {result?.length != null ? `${result.length.toFixed(3)} mm` : "—"}
                    </span>
                </div>
            </div>
            <div className="result">
                <GlomeruliCountIcon className="result-icon icon-count" />
                <div className="result-info">
                    <h2>Liczba kłębuszków</h2>
                    <span id="ilosc-klebuszkow">
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
            </div>
            <div className="result">
                <GlomeruliHealthyIcon className="result-icon icon-healthy" />
                <div className="result-info">
                    <h2>Kłębuszki niezwłóknione</h2>
                    <span className="breakdown-healthy">
                        {glomeruliBreakdown != null ? glomeruliBreakdown.healthy : "—"}
                    </span>
                </div>
            </div>
            <div className="result">
                <GlomeruliScleroticIcon className="result-icon icon-sclerotic" />
                <div className="result-info">
                    <h2>Kłębuszki zwłóknione</h2>
                    <span className="breakdown-sclerotic">
                        {glomeruliBreakdown != null ? glomeruliBreakdown.sclerotic : "—"}
                    </span>
                </div>
            </div>
            </div>
            <button className="reset-button" onClick={handleReset} aria-label="Rozpocznij nową analizę">
                <svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                    <path d="M15 9C15 12.3137 12.3137 15 9 15C5.68629 15 3 12.3137 3 9C3 5.68629 5.68629 3 9 3C10.8364 3 12.4768 3.82312 13.5962 5.125M13.5 3V5.5H11" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                Nowa analiza
            </button>
        </div>
    );
}
