import { Globe, Key, Search, Loader2 } from 'lucide-react';
import "./DataDisplay.css"

export interface DataFormProps{
    domain: string;
    setDomain: (val: string) => void;
    apiKey: string;
    setApiKey: (val: string) => void;
    loading: boolean;
    runScan: () => void;
}

export function DataDisplay({domain, setDomain, apiKey, setApiKey, loading, runScan}: DataFormProps){

    return (
        
        <div className="form-box">
            
            <div className="input-wrapper">
                <Globe className="input-icon" size={20} />
                <input
                    type="text"
                    placeholder="e.g. google.com"
                    value={domain} 
                    onChange={(e) => setDomain(e.target.value)} 
                    className="styled-input"
                />
            </div>
          

            <div className="input-wrapper">
                <Key className="input-icon" size={20} />
                <input
                    type="password"
                    placeholder="Type your API Key"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    className="styled-input"
                    autoComplete="off"
                    spellCheck="false"
                    data-1p-ignore
                    data-lpignore="true"
                    data-form-type="other"
                />
            </div>


            <div className="button-wrapper">
                <button 
                    onClick={runScan} 
                    disabled={loading}
                    className="verify-btn"
                >
                    {loading ? (
                        <>
                            <Loader2 className="spinner" size={20} /> Processing...
                        </>
                    ) : (
                        <>
                            <Search size={20} /> Verify Domain
                        </>
                    )}
                </button>
            </div>
        </div>
    )
}