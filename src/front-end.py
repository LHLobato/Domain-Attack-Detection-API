import streamlit as st
import requests
import subprocess
import sys
import time
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Detector de Domínios Maliciosos",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

URL = "http://localhost:5000/classify/domain"

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .big-font {
        font-size: 20px !important;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource  
def iniciar_api_background():
    print("🚀 Iniciando API no background...")
    

    process = subprocess.Popen(
        [sys.executable, "app.py"], 

    )
    
    time.sleep(15) 
    return process

def send_api_request(domain_name, URL=URL):
    try:
        payload = {"domain_name":domain_name}
        response = requests.post(URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            return {
                    "prediction":data.get('Prediction', 'Erro'),
                    "prob_normal":data.get('Probability_normal', 0.0),
                    "prob_malicious":data.get('Probability_malicious', 0.0),
                    "status":"success"
            }
        else:
            return {'status': 'error', 'message': f"Erro {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {'status': 'error', 'message': "Não foi possível conectar à API. Verifique se ela está rodando."}


def plot_gauge(probability):
    """Cria um gráfico de velocímetro para o nível de ameaça."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probabilidade de ser Malicioso (%)"},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkred" if probability > 0.5 else "green"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': "#e6ffe6"},
                {'range': [50, 75], 'color': "#fff2e6"},
                {'range': [75, 100], 'color': "#ffe6e6"}],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
    return fig

# --- INICIALIZAÇÃO ---

# Inicia a API automaticamente
iniciar_api_background()

# Inicializa o histórico na sessão se não existir
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- INTERFACE (SIDEBAR) ---
with st.sidebar:
    st.image("https://img.icons8.com/cloud/200/4a90e2/security-shield-green.png", width=100)
    st.title("Painel de Controle")
    st.info(
        """
        **Modelo:** BERT-Large + Random Forest + ConvNext-Nano\n
        **Backend:** Flask API\n
        **Status:** 🟢 Online\n
        """
    )
    st.markdown("---")
    st.write("🔍 **Como funciona?**")
    st.caption("O sistema analisa a semântica do nome do domínio usando NLP, features DNS utilizando tanto Machine Learning quando Visão Computacional.")


st.title("🛡️ Detector de Domínios Maliciosos")
st.markdown("### Análise de segurança baseada em Deep Learning")

col1, col2 = st.columns([3, 1])

with col1:
    domain_input = st.text_input("Digite o domínio para análise:", placeholder="exemplo: google.com", help="Insira apenas o domínio, sem http://")

with col2:
    st.write("##") # Espaçamento para alinhar o botão
    analyze_button = st.button("ANALISAR 🚀")

# Lógica de Análise
if analyze_button and domain_input:
    with st.spinner(f"Analisando '{domain_input}' com BERT..."):
        result = send_api_request(domain_input)
    
    if result['status'] == 'success':
        # Layout de Resultados
        res_col1, res_col2 = st.columns([1, 2])
        
        prob_malicious = result['prob_malicious']
        prediction = result['prediction']
        
        # Adiciona ao histórico
        st.session_state['history'].insert(0, {
            "Domínio": domain_input,
            "Predição": prediction,
            "Risco": f"{prob_malicious*100:.1f}%",
            "Hora": time.strftime("%H:%M:%S")
        })

        with res_col1:
            st.subheader("Veredito")
            if prediction == 1 or prediction == "Malicious": # Ajuste conforme o retorno da sua API
                st.error("🚨 MALICIOSO DETECTADO")
                st.markdown(f"**Nível de Confiança:** {prob_malicious*100:.2f}%")
            else:
                st.success("✅ DOMÍNIO SEGURO")
                st.markdown(f"**Probabilidade de ser Seguro:** {result['prob_normal']*100:.2f}%")
        
        with res_col2:
            # Mostra o gráfico
            st.plotly_chart(plot_gauge(prob_malicious), use_container_width=True)

    elif result['status'] == 'error':
        st.error(result['message'])

# --- HISTÓRICO DE ANÁLISES ---
st.markdown("---")
st.subheader("📜 Histórico Recente")

if st.session_state['history']:
    df_history = pd.DataFrame(st.session_state['history'])
    
    # Função para colorir a tabela
    def highlight_risk(val):
        color = '#ffcccc' if 'Malicious' in str(val) or '1' in str(val) else '#ccffcc'
        return f'background-color: {color}'

    st.dataframe(
        df_history, 
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Nenhuma análise realizada ainda nesta sessão.")