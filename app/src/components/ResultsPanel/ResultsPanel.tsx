import './ResultsPanel.css'

export default function ResultsPanel() {
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
                    <span id="zwloknienie">0.0%</span>
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
                    <span id="dlugosc">0.0mm</span>
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
                    <span id="ilosc-klebuszkow">0</span>
                </div>
            </div>
        </div> 
    );
}