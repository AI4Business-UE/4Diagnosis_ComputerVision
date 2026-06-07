import './ResultsPanel.css'

interface ResultsPanelProps {
    result: {
        length?: number;
        fibrosis_ratio?: number;
        glomeruli_count?: number;
    } | null;
    glomeruliScanning?: boolean;
    glomeruliBreakdown?: { healthy: number; sclerotic: number };
}


function plPL(n: number, one: string, few: string, many: string): string {
    if (n === 1) return one;
    const mod10 = n % 10;
    const mod100 = n % 100;
    if (mod10 >= 2 && mod10 <= 4 && (mod100 < 10 || mod100 >= 20)) return few;
    return many;
}

export default function ResultsPanel({ result, glomeruliScanning, glomeruliBreakdown }: ResultsPanelProps) {
    const displayCount = glomeruliBreakdown != null
        ? glomeruliBreakdown.healthy + glomeruliBreakdown.sclerotic
        : result?.glomeruli_count;

    return (
        <div className="results-panel">
            <p>
                Wyniki analiz
            </p>
            <div className="result">
                <img
                    src={"/chart.svg"}
                    width={28}
                    height={28}
                    alt=""
                    aria-hidden="true"
                    className="icon-chart"
                />
                <div className="result-info">
                    <h2>Procent zwłóknienia</h2>
                    <span id="zwloknienie">
                        {result?.fibrosis_ratio != null
                            ? `${(result.fibrosis_ratio * 100).toFixed(2)}%`
                            : "—"}
                    </span>
                </div>
            </div>
            <div className="result">
                <img
                    src={"/ruler.svg"}
                    width={28}
                    height={28}
                    alt=""
                    aria-hidden="true"
                    className="icon-ruler"
                />
                <div className="result-info">
                    <h2>Długość tkanki</h2>
                    <span>
                        {result?.length != null
                            ? `${result.length.toFixed(3)} mm`
                            : "—"}
                    </span>
                </div>
            </div>
            <div className="result">
                <img
                    src={"/circle.svg"}
                    width={28}
                    height={28}
                    alt=""
                    aria-hidden="true"
                    className="icon-circle"
                />
                 <div className="result-info">
                    <h2>Liczba kłębuszków</h2>
                    <span id="ilosc-klebuszkow">
                        {displayCount != null
                            ? <>
                                {displayCount}
                                {glomeruliScanning && <span className="scanning-indicator"> skanowanie...</span>}
                                {glomeruliBreakdown != null && (
                                    <span className="glomeruli-breakdown">
                                        <span className="breakdown-healthy">
                                            {glomeruliBreakdown.healthy} {plPL(glomeruliBreakdown.healthy, 'niezwłókniony', 'niezwłóknione', 'niezwłóknionych')}
                                        </span>
                                        <span className="breakdown-sclerotic">
                                            {glomeruliBreakdown.sclerotic} {plPL(glomeruliBreakdown.sclerotic, 'zwłókniony', 'zwłóknione', 'zwłóknionych')}
                                        </span>
                                    </span>
                                )}
                              </>
                            : "—"}
                    </span>
                </div>
            </div>
        </div>
    );
}
