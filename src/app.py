from flask import Flask, jsonify, request
from ensemble import Domain_Ensemble
import numpy as np 
import sys
import re
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from pyts.image import GramianAngularField
import torch
import joblib
import dns.resolver
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)
# --- CONSTANTES ---
DOMAIN_FEATURE_ORDER = [
    "A Count", "IP Count", "MX Count", "NS Count", "A Missing", "ASN Count", 
    "SOA Retry", "AAAA Count", "MX Missing", "A TTL Count", "CNAME Count", 
    "SOA Minimum", "SOA Missing", "SOA Refresh", "A Medium TTL", "MX Medium TTL", 
    "SOA TTL Count", "SOA Medium TTL", "AAAA Medium TTL", "SOA Expire Length", 
    "SOA Serial Length", "A TTL Standard Deviation", "SOA TTL Standard Deviation"
]

# --- CARREGAMENTO DE MODELOS ---
try:
    scaler = joblib.load("../models/scaler.joblib")
    selector = joblib.load("../models/selector.joblib")
    vectorizer = joblib.load("../models/tfidf.joblib")

    classifier = Domain_Ensemble.load_ensemble("../models/", device="cpu")
    print("✅ Ensemble correctly initialized!")
except Exception as e:
    print(f"❌ Erro ao carregar modelos: {e}")
    sys.exit(1)


# --- FUNÇÕES DE EXTRAÇÃO (DNS LIVE) ---
def extract_live_dns_features(domain):
    features = {k: 0.0 for k in DOMAIN_FEATURE_ORDER}
    
    resolver = dns.resolver.Resolver()
    resolver.timeout = 2
    resolver.lifetime = 2


    def calculate_ttl_stats(answers):
        if not answers:
            return 0, 0, 0
        
        try:
            ttl = float(answers.rrset.ttl)
        except AttributeError:

            ttl = float(answers.ttl) if hasattr(answers, 'ttl') else 0.0
            
        count = len(answers)
        mean = ttl
        std = 0.0 
        
        return count, mean, std
    # ---------------------

    # 1. Consulta A
    try:
        a_records = resolver.resolve(domain, 'A')
        features['A Count'] = len(a_records)
        features['IP Count'] = len(a_records)
        features['A Missing'] = 0
        count, mean, std = calculate_ttl_stats(a_records)
        features['A TTL Count'] = count
        features['A Medium TTL'] = mean
        features['A TTL Standard Deviation'] = std
        features['ASN Count'] = 1 
    except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN, dns.exception.Timeout):
        features['A Missing'] = 1

    try:
        aaaa_records = resolver.resolve(domain, 'AAAA')
        features['AAAA Count'] = len(aaaa_records)
        _, mean, _ = calculate_ttl_stats(aaaa_records)
        features['AAAA Medium TTL'] = mean
    except: pass

    try:
        mx_records = resolver.resolve(domain, 'MX')
        features['MX Count'] = len(mx_records)
        features['MX Missing'] = 0
        _, mean, _ = calculate_ttl_stats(mx_records)
        features['MX Medium TTL'] = mean
    except: features['MX Missing'] = 1


    try:
        ns_records = resolver.resolve(domain, 'NS')
        features['NS Count'] = len(ns_records)
    except: pass


    try:
        cname_records = resolver.resolve(domain, 'CNAME')
        features['CNAME Count'] = len(cname_records)
    except: pass


    soa_found = False
    current_domain_search = domain
    
    # Loop para subir a hierarquia (ex: a.b.c.com -> b.c.com -> c.com)
    # Limite de 3 tentativas para não ficar preso
    for _ in range(3):
        try:
            soa_records = resolver.resolve(current_domain_search, 'SOA')
            
            # Se chegou aqui, achou!
            soa = soa_records[0]
            features['SOA Missing'] = 0
            
            features['SOA Retry'] = float(soa.retry)
            features['SOA Refresh'] = float(soa.refresh)
            features['SOA Minimum'] = float(soa.minimum)
            features['SOA Expire Length'] = float(soa.expire)
            features['SOA Serial Length'] = len(str(soa.serial))
            
            _, mean, std = calculate_ttl_stats(soa_records)
            features['SOA TTL Count'] = 1
            features['SOA Medium TTL'] = mean
            features['SOA TTL Standard Deviation'] = std
            
            soa_found = True
            break # Pare de procurar
            
        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN):
            # Se não achou, remove a primeira parte e tenta o pai
            # Ex: mail.google.com -> google.com
            parts = current_domain_search.split('.')
            if len(parts) > 2:
                current_domain_search = ".".join(parts[1:])
            else:
                break # Chegou na raiz e não achou
        except:
            break

    if not soa_found:
        features['SOA Missing'] = 1

    return features

def shannon_entropy(s: str) -> float:
    if not s: return 0.0
    _, counts = np.unique(list(s), return_counts=True)
    prob = counts / len(s)
    return -np.sum(prob * np.log2(prob))

def vowel_ratio(s: str) -> float:
    s = re.sub(r'[^a-z]', '', s.lower())
    if not s: return 0.0
    vowels = sum(ch in 'aeiou' for ch in s)
    return vowels / len(s) if len(s) > 0 else 0.0

def digit_ratio(s: str) -> float:
    if not s: return 0.0
    digits = sum(ch.isdigit() for ch in s)
    return digits / len(s) if len(s) > 0 else 0.0

def consonant_ratio(s: str) -> float:
    s = re.sub(r'[^a-z]', '', s.lower())
    if not s: return 0.0
    consonants = sum(ch not in 'aeiou' for ch in s)
    return consonants / len(s) if len(s) > 0 else 0.0

def special_char_ratio(s: str) -> float:
    if not s: return 0.0
    specials = sum(ch in '-_.' for ch in s)
    return specials / len(s) if len(s) > 0 else 0.0

def extract_lexical_features(domain: str):
    domain_original = str(domain).lower().strip() 
    if not domain_original:
        return None
        
    domain_no_www = re.sub(r'^www\.', '', domain_original)
    parts = domain_no_www.split('.')
    main = parts[-2] if len(parts) > 1 else parts[0]

    consonants = re.findall(r'[^aeiou\d\W_]+', domain_original)
    max_consonant_seq = max(len(s) for s in consonants) if consonants else 0
    digits_seq = re.findall(r'\d+', domain_original)
    max_digit_seq = max(len(s) for s in digits_seq) if digits_seq else 0
    unique_char_ratio = len(set(domain_original)) / len(domain_original)

    return [len(domain_original),
        len(main),
        len(parts) - 2 if len(parts) > 2 else 0,
        sum(c.isdigit() for c in domain_original),
        digit_ratio(domain_original),
        vowel_ratio(domain_original),
       consonant_ratio(domain_original),
        special_char_ratio(domain_original),
        shannon_entropy(domain_original),
        domain_original.count('-'),
        1 if domain_original[0].isdigit() else 0,
        1 if re.search(r'(.)\1{2,}', domain_original) else 0,
        max_consonant_seq,
        max_digit_seq,
       round(unique_char_ratio, 4)]

    
@app.route('/classify/domain', methods=['POST'])
@limiter.limit("1/second", override_defaults=False)
def inference():
    if not all([classifier, scaler, vectorizer, selector]):
        return jsonify({"error": "Models are not available on the server."}), 503
    
    input_data = request.get_json()
    
    if not input_data:
        return jsonify({"error": "Invalid Request"}), 400
    
    try:
        name = input_data['domain_name']
        lexical_features = extract_lexical_features(name)
        dns_live = extract_live_dns_features(name)

        dns_values = [dns_live[feature] for feature in DOMAIN_FEATURE_ORDER]

        vectorized_name = vectorizer.transform([name]).toarray()
        dns_scaled = scaler.transform([dns_values])

        image_features = np.hstack((vectorized_name, dns_scaled))
        tabular_features = np.hstack((dns_values, lexical_features)).reshape(1, -1)

        image_features = selector.transform(image_features)

        gaf = GramianAngularField(method="summation")
        image = gaf.transform(image_features)

        cmap = cm.get_cmap('rainbow') 

        norm = Normalize(vmin=-1, vmax=1)
        processed_image = cmap(norm(image))
        image_f16 = processed_image[..., :3]
            
        image_rgb_transposed = image_f16.transpose(0, 3, 1, 2).astype(np.float32)
        image_rgb_transposed = torch.tensor(image_rgb_transposed, dtype=torch.float32)

        y_pred_proba = classifier.predict_proba(image_rgb_transposed, tabular_features, [name])[:, 1]
        y_pred = classifier.predict(image_rgb_transposed, tabular_features, [name])
        
        is_malicious = float(y_pred_proba[0])
        response = {
            "Prediction": "Malicious" if y_pred[0] == 1 else "Normal",
            "Probability_malicious" : is_malicious,
            "Probability_normal" : 1 - is_malicious
        }

        return jsonify(response), 200
    except Exception as e:
            print(f"Erro na inferência: {e}")
            return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0', debug=True)