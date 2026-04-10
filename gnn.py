def predict_url(url):
    # 1. Load model
    loaded_model = joblib.load("opqr_model.pkl")
    
    # 2. Extract features from URL
    feat = extract_features(url)
    feat_df = pd.DataFrame([feat])
    
    # 3. Predict
    prob = loaded_model.predict_proba(feat_df)[0]  # [benign prob, phishing prob]
    result = loaded_model.predict(feat_df)[0]
    
    print(f"\nTarget URL: {url}")
    print(f"result: {'malicious' if result == 1 else 'safe'}")
    print(f"probability: {prob[1]*100:.2f}%")

# Test run
predict_url("http://secure-login-naver.com-check.top/login")
predict_url("https://github.com/trending")









import requests

def get_redirect_chain(url):
    chain = [url]
    try:
        # Follow redirects to the end
        response = requests.get(url, allow_redirects=True, timeout=5)
        
        if response.history:
            for resp in response.history:
                chain.append(resp.url)
        
        if response.url not in chain:
            chain.append(response.url)
            
        print(f"Redirect chain: {' -> '.join(chain)}")
        return chain
    except Exception as e:
        print(f"Redirect trace failed: {e}")
        return chain

# Example
test_url = "https://www.naver.com"
path = get_redirect_chain(test_url)









def integrated_analysis(url):
    # 1. Collect per-model scores (0.0–1.0)
    rf_score = rf_model.predict_proba(extract_features(url))[0][1]
    gnn_score = get_gnn_score(url)
    kobert_score = get_kobert_score(url)

    # 2. Weighted blend (example)
    final_score = (rf_score * 0.2) + (kobert_score * 0.3) + (gnn_score * 0.5)

    # 3. Threshold to reduce false positives
    is_malicious = final_score > 0.7
    
    return is_malicious, final_score









import joblib
import os

file_name = "opqr_model.pkl"

if os.path.exists(file_name):
    rf_model = joblib.load(file_name)
    print(f"Loaded {file_name} OK")
else:
    print(f"Missing {file_name}; check path or re-save the model.")










from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("opqr_model.pkl")

def extract_features(url):
    # Length, dots, special chars, etc.
    # ... plug in your training-time feature builder ...
    return features_list

@app.get("/check")
def check_url(url: str):
    features = extract_features(url)
    prob = model.predict_proba([features])[0][1]
    
    return {
        "url": url,
        "is_malicious": prob > 0.7,
        "probability": f"{prob*100:.2f}%"
    }










import networkx as nx

def analyze_relation(graph, start_node):
    print(f"\n[{start_node}] tracing related paths...")
    
    paths = list(nx.dfs_edges(graph, source=start_node))
    
    if not paths:
        print("-> No other sites linked (single node).")
        return

    is_dangerous_path = False
    
    for u, v in paths:
        target_status = graph.nodes[v].get('label', 'UNKNOWN')
        print(f"-> {u} -> {v} [status: {target_status}]")
        
        if target_status == "MALICIOUS":
            is_dangerous_path = True

    print("-" * 40)
    if is_dangerous_path:
        print(f"Result: [{start_node}] may look SAFE alone, but...")










import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore")

try:
    model = joblib.load("opqr_model.pkl")
except Exception:
    print("Could not load model file.")

def get_features(url):
    digit_count = sum(c.isdigit() for c in url)
    features = [
        url.count('.'),                                      
        url.count('-'),                                      
        len(url),                                            
        digit_count / len(url) if len(url) > 0 else 0,       
        1 if url.startswith("https") else 0,                 
        1 if any(word in url for word in ["login", "verify", "account"]) else 0, 
        url.count('/')                                       
    ]
    return np.array(features).reshape(1, -1)

input_url = input("Enter URL to analyze: ")

try:
    data = get_features(input_url)
    prob = model.predict_proba(data)[0][1]
    
    result_label = "MALICIOUS" if prob > 0.7 else "SAFE"
    
    print("\n" + " [ Analysis report ] ".center(46, "-"))
    print(f"  > URL: {input_url}")
    print(f"  > Verdict: {result_label}")
    print(f"  > Risk probability: {prob*100:.2f}%")
    print("-" * 50)
    print("  Status: Analysis completed.")

except Exception as e:
    print(f"Analysis error: {e}")
