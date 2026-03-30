import './ResultsPanel.css'

interface ResultsPanelProps {
    result: {
        length?: number;
        fibrosis_ratio?: number;
        glomeruli_count?: number;
    } | null;
}


export default function ResultsPanel({ result }: ResultsPanelProps) {
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
                        {result?.glomeruli_count ?? "-"}
                    </span>
                </div>
            </div>
        </div>
    );
}