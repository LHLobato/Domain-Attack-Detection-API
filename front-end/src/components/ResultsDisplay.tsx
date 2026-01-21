import { AlertTriangle, ShieldAlert, ShieldCheck } from "lucide-react";
import type { Domain_Classification } from "../api";
import "./ResultsDisplay.css"


interface ResultsDisplayProps{
    error:string;
    result: Domain_Classification|null 
}

export function ResultsDisplay({error, result} : ResultsDisplayProps){

    if(!result && !error) return null;

    const isMalicious = result?.Prediction === 'Malicious';
    const statusClass = isMalicious ? 'danger': 'safe';

    const maliciousPct = result ? (result.Probability_malicious *100): 0;
    const normalPct = result ? (result.Probability_normal * 100) : 0;

    return (
        <div className="results-container">
            {error && (
                <div className="error-box">
                    <AlertTriangle size={24}/>
                    <p>{error}</p>
                </div>
            )}

            {result && (

                <div className={`result-card ${statusClass}`}>
                    <div className="card-header">
                        <div className="icon-box">
                            {isMalicious ? <ShieldAlert size={32} /> : <ShieldCheck size={32} />}
                            <h3>{isMalicious ? "Ameaça Detectada, Domínio possivelmente Malicioso" : "Domínio Provavelmente Seguro"}</h3>
                            </div>
                    </div>

                    <div className="stats-grid">
                        
                        <div className="stat-item">
                            <div className="stat-label">
                                <span>Probabilidade de Malicioso</span>
                                <strong>{maliciousPct.toFixed(2)}%</strong>
                            </div>
                            <div className="progress-bg">
                                <div 
                                    className="progress-fill fill-danger" 
                                    style={{ width: `${maliciousPct}%` }}
                                ></div>
                            </div>
                        </div>

                        <div className="stat-item">
                            <div className="stat-label">
                                <span>Probabilidade de Legítimo</span>
                                <strong>{normalPct.toFixed(2)}%</strong>
                            </div>
                            <div className="progress-bg">
                                <div 
                                    className="progress-fill fill-safe" 
                                    style={{ width: `${normalPct}%` }}
                                ></div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
    
        </div>
    )
}