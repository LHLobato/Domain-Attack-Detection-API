import { useScanner } from '../useScanner'; 
import './HomePage.css'; 
import { ResultsDisplay } from './ResultsDisplay';
import { DataDisplay } from './DataDisplay';

export function HomePage() {

  const { 
    domain, setDomain, 
    apiKey, setApiKey, 
    result, loading, error, runScan 
  } = useScanner();

  return (

      <div className="home-container"> 
        
        <h1>Malicious Domain Detector</h1>
        
        <DataDisplay 
            domain={domain} 
            setDomain={setDomain} 
            apiKey={apiKey} 
            setApiKey={setApiKey} 
            loading={loading} 
            runScan={runScan}   
        />

        <ResultsDisplay error={error} result={result} />
      </div>
  );
}