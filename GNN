def predict_url(url):
    # 1. 모델 불러오기
    loaded_model = joblib.load("opqr_model.pkl")
    
    # 2. 입력받은 URL에서 특징 추출
    feat = extract_features(url)
    feat_df = pd.DataFrame([feat])
    
    # 3. 예측
    prob = loaded_model.predict_proba(feat_df)[0] # [정상 확률, 피싱 확률]
    result = loaded_model.predict(feat_df)[0]
    
    print(f"\n대상 URL: {url}")
    print(f"분석 결과: {'⚠️ 피싱 위험' if result == 1 else '✅ 안전'}")
    print(f"위험 확률: {prob[1]*100:.2f}%")

# 테스트 실행
predict_url("http://secure-login-naver.com-check.top/login")
predict_url("https://github.com/trending")









import requests

def get_redirect_chain(url):
    chain = [url]
    try:
        # allow_redirects=True로 설정하여 끝까지 추적
        response = requests.get(url, allow_redirects=True, timeout=5)
        
        # 중간에 거쳐간 경로들 파악
        if response.history:
            for resp in response.history:
                chain.append(resp.url)
        
        # 최종 목적지 추가 (이미 추가되지 않았다면)
        if response.url not in chain:
            chain.append(response.url)
            
        print(f"🔗 리다이렉트 경로 발견: {' -> '.join(chain)}")
        return chain
    except Exception as e:
        print(f"⚠️ 추적 실패: {e}")
        return chain

# 테스트 해보기
test_url = "https://www.naver.com" # 실제 단축 URL 예시
path = get_redirect_chain(test_url)









def integrated_analysis(url):
    # 1. 각 모델의 확률값(0.0 ~ 1.0) 수집
    rf_score = rf_model.predict_proba(extract_features(url))[0][1]
    gnn_score = get_gnn_score(url) # 그래프 기반 점수
    kobert_score = get_kobert_score(url) # 맥락 기반 점수

    # 2. 오탐 방지를 위한 가중치 적용 (예시)
    # GNN은 확실할 때만 점수가 높으므로 가중치를 높게 설정
    final_score = (rf_score * 0.2) + (kobert_score * 0.3) + (gnn_score * 0.5)

    # 3. 임계값(Threshold) 설정으로 오탐 조절
    # 0.5가 아닌 0.7 정도로 높이면 확실한 것만 '위험'으로 분류 (오탐 감소)
    is_malicious = final_score > 0.7
    
    return is_malicious, final_score









import joblib
import os

# 파일이 진짜 있는지 확인부터 해봅시다
file_name = "opqr_model.pkl"

if os.path.exists(file_name):
    rf_model = joblib.load(file_name)
    print(f"✅ {file_name} 로드 성공!")
else:
    print(f"❌ {file_name} 파일이 없습니다. 저장된 이름을 확인하거나 다시 저장하세요.")










from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# 1. 모델 불러오기
model = joblib.load("opqr_model.pkl")

# 2. 특징 추출 함수 (아까 만든 것과 동일해야 함)
def extract_features(url):
    # 길이, 점 개수, 특수문자 등 수치화 로직
    # ... (기존에 정의한 함수 코드 입력) ...
    return features_list

@app.get("/check")
def check_url(url: str):
    # 특징 추출 및 데이터프레임 변환
    features = extract_features(url)
    
    # 모델 예측 (확률값 추출)
    prob = model.predict_proba([features])[0][1] # 피싱일 확률
    
    return {
        "url": url,
        "is_malicious": prob > 0.7,  # 70% 이상이면 위험으로 간주
        "probability": f"{prob*100:.2f}%"
    }










import networkx as nx

# 4. 연관성 분석 함수 (완성 버전)
def analyze_relation(graph, start_node):
    print(f"\n🔍 [{start_node}] 연관 경로 추적 시작...")
    
    # 1) 연결된 모든 경로(Edges) 가져오기 (깊이 우선 탐색)
    paths = list(nx.dfs_edges(graph, source=start_node))
    
    if not paths:
        print("-> 연결된 다른 사이트가 없습니다. (단일 노드)")
        return

    is_dangerous_path = False
    
    # 2) 경로를 따라가며 연결된 노드들의 상태 확인
    for u, v in paths:
        target_status = graph.nodes[v].get('label', 'UNKNOWN')
        print(f"-> {u} 🔗 {v} [상태: {target_status}]")
        
        # 연결된 노드 중 하나라도 악성(MALICIOUS)이 있으면 위험 경로로 간주
        if target_status == "MALICIOUS":
            is_dangerous_path = True

    # 3) 최종 연관성 판정
    print("-" * 40)
    if is_dangerous_path:
        print(f"🚨 결과: [{start_node}] 자체는 SAFE해 보일 수 있으나,")










import joblib
import numpy as np
import warnings

# 1. 모든 경고 메시지 차단
warnings.filterwarnings("ignore")

# 2. 모델 로드
try:
    model = joblib.load("opqr_model.pkl")
except:
    print("❌ 모델 파일을 불러올 수 없습니다.")

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

# 3. 실시간 인터페이스 (타이틀 제거 버전)
input_url = input("🔍 분석할 URL을 입력하세요: ")

try:
    data = get_features(input_url)
    prob = model.predict_proba(data)[0][1]
    
    result_label = "🚨 위험 (MALICIOUS)" if prob > 0.7 else "✅ 안전 (SAFE)"
    
    print("\n" + " [ 분석 결과 리포트 ] ".center(46, "-"))
    print(f"  > 분석 대상: {input_url}")
    print(f"  > 탐지 결과: {result_label}")
    print(f"  > 위험 확률: {prob*100:.2f}%")
    print("-" * 50)
    print("  Status: Analysis Completed.")

except Exception as e:
    print(f"❌ 분석 오류: {e}")
