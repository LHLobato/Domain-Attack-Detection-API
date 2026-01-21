import axios, { type AxiosResponse } from 'axios';

const URL_BASE = "https://luish2009-domain-api.hf.space"

interface Domain{
    domain_name: string;
}

export interface Domain_Classification{
    Prediction: string;
    Probability_malicious: number; 
    Probability_normal: number;
}

export async function getDomainClassification(domainName: string, apiKey:string): Promise<Domain_Classification> {

    const payload: Domain = {
        domain_name: domainName
    };


    try{
        const response: AxiosResponse<Domain_Classification>  = await axios.post(`${URL_BASE}/classify/domain`,
            payload, 
            {
                headers: {
                'Content-Type': 'application/json',
                'X-Api-Key': apiKey 
            }
            }
        );
        return response.data 
    } catch(error){
        console.error('Error at catching domain classification', error);
        throw error;
    }
}