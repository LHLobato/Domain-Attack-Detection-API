import { BookOpen, Terminal, Shield, Server, Code } from 'lucide-react';
import './DocPage.css';

export function DocPage() {
    return (
        <div className="doc-container">
            
            <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
                <h1 style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>
                    API Documentation <BookOpen size={32} style={{ verticalAlign: 'middle', marginLeft: '10px', color: '#00ff88' }} />
                </h1>
                <p style={{ fontSize: '1.1rem', color: '#888' }}>
                    Integration guide for the malicious domain detection system.
                </p>
            </div>

            <section className="doc-section">
                <h2><Shield size={24} /> About the API</h2>
                <p>
                    The <strong>Domain Scanner API</strong> uses an ensemble of Deep Learning models to analyze domains in real-time.
                    It extracts lexical features and performs live DNS lookups to determine the probability of a domain being malicious (phishing, malware, etc.).
                </p>
            </section>

            <section className="doc-section">
                <h2><Server size={24} /> Endpoint</h2>
                <div className="code-card">
                    <p style={{ margin: 0, fontSize: '1.1rem', fontFamily: 'monospace' }}>
                        <span className="method">POST</span> 
                        https://luish2009-domain-api.hf.space/classify/domain
                        <span className="endpoint-badge">HTTPS</span>
                    </p>
                </div>
            </section>

            <section className="doc-section">
                <h2><Code size={24} /> Authentication</h2>
                <p>
                    All requests must include the <code>X-Api-Key</code> header with your private access token.
                </p>
                <div className="code-card">
                    <pre>
                        Headers:<br/>
                        <span className="key">Content-Type:</span> application/json<br/>
                        <span className="key">X-Api-Key:</span> YOUR_TOKEN_HERE
                    </pre>
                </div>
            </section>

            <section className="doc-section">
                <h2><Terminal size={24} /> Usage Example (cURL)</h2>
                <p>Send the domain you wish to analyze in the request body (JSON).</p>
                
                <div className="code-card">
                    <pre>
<span className="cmd">curl</span> -X POST https://luish2009-domain-api.hf.space/classify/domain \<br/>
  -H <span className="string">"Content-Type: application/json"</span> \<br/>
  -H <span className="string">"X-Api-Key: 12345"</span> \<br/>
  -d <span className="string">'&#123;"domain_name": "google.com"&#125;'</span>
                    </pre>
                </div>
            </section>

            <section className="doc-section">
                <h2><Code size={24} /> Response (JSON)</h2>
                <p>The API returns the final prediction and the probabilities calculated by the model.</p>
                
                <div className="code-card">
                    <pre>
&#123;<br/>
  <span className="key">"Prediction"</span>: <span className="string">"Normal"</span>,<br/>
  <span className="key">"Probability_malicious"</span>: <span className="number">0.0012</span>,<br/>
  <span className="key">"Probability_normal"</span>: <span className="number">0.9988</span><br/>
&#125;
                    </pre>
                </div>
            </section>

        </div>
    );
}