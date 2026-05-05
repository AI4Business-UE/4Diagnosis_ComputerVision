import './ResultsPanel.css'

interface ResultsPanelProps {
    result: {
        length?: number;
        fibrosis_ratio?: number;
        glomeruli_count?: number;
    } | null;
}
const handleReset = () => {
    window.location.reload();
};

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
        <button 
            className="reset-button" 
            onClick={handleReset}
            aria-label="Rozpocznij nową analizę"
        >
            <svg 
                width="18" 
                height="18" 
                viewBox="0 0 18 18" 
                fill="none" 
                xmlns="http://www.w3.org/2000/svg"
                aria-hidden="true"
            >
                <path 
                    d="M15 9C15 12.3137 12.3137 15 9 15C5.68629 15 3 12.3137 3 9C3 5.68629 5.68629 3 9 3C10.8364 3 12.4768 3.82312 13.5962 5.125M13.5 3V5.5H11" 
                    stroke="currentColor" 
                    strokeWidth="1.5" 
                    strokeLinecap="round" 
                    strokeLinejoin="round"
                />
            </svg>
            Nowa analiza
        </button>    
        </div>
        
    );
}