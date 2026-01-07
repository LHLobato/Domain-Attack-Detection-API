import requests
import json
import time


URL = "http://localhost:5000/classify/domain"

safe_domains = [
    {"name": "google.com", "desc": "O gigante das buscas"},
    {"name": "facebook.com", "desc": "Rede Social (Meta)"},
    {"name": "youtube.com", "desc": "Plataforma de Vídeo (Google)"},
    {"name": "twitch.tv", "desc": "Streaming de Games (Amazon)"},
    {"name": "wikipedia.org", "desc": "Enciclopédia livre"},
    {"name": "python.org", "desc": "Linguagem de programação (Nossa mãe)"},
    {"name": "github.com", "desc": "Repositório de código"},
    {"name": "unesp.br", "desc": "UNESP (Universidade Estadual Paulista)"},
    {"name": "usp.br", "desc": "USP (Universidade de São Paulo)"},
    {"name": "amazon.com", "desc": "E-commerce global"},
    {"name": "microsoft.com", "desc": "Tech Corporativa"},
    {"name": "nasa.gov", "desc": "Governo Americano (Espaço)"},
    {"name": "stackoverflow.com", "desc": "Onde a gente resolve os bugs"},
    {"name": "bbc.co.uk", "desc": "Notícias internacionais"}
]

print(f"🛡️  Iniciando Teste de Sanidade Estendido (Apenas Benignos)...")
print(f"🎯 Alvo: {URL}\n")

for i, domain in enumerate(safe_domains, 1):
    d_name = domain['name']
    d_desc = domain['desc']
    
    print(f"▶️  Teste {i}/{len(safe_domains)}: {d_name}")
    print(f"    Contexto: {d_desc}")
    
    start_time = time.time()
    
    try:
        # Envia apenas o nome para a API fazer a mágica do DNS ao vivo
        payload = {"domain_name": d_name}
        response = requests.post(URL, json=payload)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            prediction = data.get('Prediction', 'Erro')
            prob_normal = data.get('Probability_normal', 0.0)
            prob_malicious = data.get('Probability_malicious', 0.0)
            
            # Ícones de status
            if prediction == "Normal":
                icon = "✅ CORRETO"
            else:
                icon = "🤔 ESTRANHO (Falso Positivo)"
            
            print(f"    ⏱️  Tempo de Resposta: {elapsed:.3f}s")
            print(f"    📊 Classificação: {icon}")
            print(f"    🕊️  Probabilidade Benigna: {prob_normal:.2%}")
            
            # Se deu malicioso, mostra o quanto ele achou que era (para debug)
            if prediction == "Malicious":
                print(f"    💀 Probabilidade Maliciosa: {prob_malicious:.2%}")
                
        else:
            print(f"    ⚠️  Erro HTTP {response.status_code}: {response.text}")

    except requests.exceptions.ConnectionError:
        print("    ❌ Erro: API fora do ar. Verifique se o app.py está rodando.")
    except Exception as e:
        print(f"    ❌ Erro Inesperado: {e}")
    
    print("-" * 50)
