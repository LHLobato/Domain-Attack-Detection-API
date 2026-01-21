import { useState } from 'react'
import { getDomainClassification } from './api'

export function useScanner(){

    const[domain, setDomain] = useState('');
    const[apiKey, setApiKey] = useState('');
    const[result, setResult] = useState<any>(null);
    const[loading, setLoading] = useState(false);
    const[error, setError] = useState('');


    async function runScan(){
        if(!domain || !apiKey){
            setError("Remaining data");
            return; 
        }
        
        setLoading(true);
        setError('');
        setResult(null);

        try{
            const data = await getDomainClassification(domain, apiKey);
            setResult(data);
        }catch(err){
            setError("Probably API Key not confirmed.")
        } finally{
            setLoading(false);
        }
    }

    return {domain, setDomain, 
        apiKey, setApiKey, 
        result, 
        loading, 
        error, 
        runScan
    };
}