import requests
import pandas as pd
import os

# 1. 저장용 폴더 생성
os.makedirs("Blacklist", exist_ok=True)
os.makedirs("Whitelist", exist_ok=True)

def collect_with_source(target=500):
    all_data = []
    
    print("🚀 [블랙리스트] 출처별 수집 시작...")
    
    # --- 소스 1: OpenPhish ---
    try:
        res = requests.get("https://openphish.com/feed.txt", timeout=5)
        urls = [u.strip() for u in res.text.split('\n') if u.strip()]
        for u in urls:
            if len(all_data) < target:
                all_data.append({"url": u, "label": 1, "source": "OpenPhish"})
        print(f"✅ OpenPhish 수집 완료 (누적: {len(all_data)})")
    except: print("⚠️ OpenPhish 실패")

    # --- 소스 2: URLhaus ---
    if len(all_data) < target:
        try:
            res = requests.get("https://urlhaus.abuse.ch/downloads/text/", timeout=5)
            urls = [u.strip() for u in res.text.split('\n') if u.strip() and not u.startswith('#')]
            for u in urls:
                if len(all_data) < target:
                    all_data.append({"url": u, "label": 1, "source": "URLhaus"})
            print(f"✅ URLhaus 추가 완료 (누적: {len(all_data)})")
        except: print("⚠️ URLhaus 실패")

    # --- 화이트리스트 생성 ---
    print("🚀 [화이트리스트] 생성 시작...")
    seeds = ["google.com", "naver.com", "daum.net", "kakao.com", "youtube.com", "apple.com", "github.com"]
    for i in range(target):
        domain = seeds[i % len(seeds)]
        all_data.append({"url": domain, "label": 0, "source": "Tranco_Top_List"})

    # 데이터프레임 생성
    df = pd.DataFrame(all_data)
    
    # 2. 파일 분리 저장
    black_df = df[df['label'] == 1]
    white_df = df[df['label'] == 0]
    
    black_df.to_csv("Blacklist/blacklist_with_source.csv", index=False)
    white_df.to_csv("Whitelist/whitelist_with_source.csv", index=False)
    df.to_csv("gnn_total_dataset.csv", index=False)
    
    return df

# 실행
total_df = collect_with_source(500)

print("\n" + "="*45)
print("📂 폴더 분류 완료!")
print("- Blacklist/blacklist_with_source.csv (출처 포함)")
print("- Whitelist/whitelist_with_source.csv (출처 포함)")
print(f"총 데이터: {len(total_df)}개")
print("="*45)

# 출처별 개수 확인
print("\n🔍 [출처별 데이터 분포]")
print(total_df['source'].value_counts())


import numpy as np

def extract_features(df):
    print("🧠 특징 추출 시작...")
    
    # 1. URL 길이
    df['url_len'] = df['url'].apply(len)
    
    # 2. 특수문자 개수 추출
    df['count_dot'] = df['url'].apply(lambda x: x.count('.'))
    df['count_hyphen'] = df['url'].apply(lambda x: x.count('-'))
    df['count_at'] = df['url'].apply(lambda x: x.count('@'))
    df['count_slash'] = df['url'].apply(lambda x: x.count('/'))
    
    # 3. 숫자가 포함된 비율
    def digit_count(url):
        digits = [i for i in url if i.isdigit()]
        return len(digits) / len(url)
    df['digit_ratio'] = df['url'].apply(digit_count)
    
    # 4. HTTPS 사용 여부 (1: 사용, 0: 미사용)
    df['is_https'] = df['url'].apply(lambda x: 1 if "https" in x else 0)
    
    print("✅ 특징 추출 완료!")
    return df

# 특징 추출 실행
fe_df = extract_features(total_df)

# 결과 확인
print(fe_df.head())









import re

def extract_features(url):
    features = {}
    
    # 1. 길이 관련 특징
    features['url_len'] = len(url)
    features['dot_count'] = url.count('.')
    features['hyphen_count'] = url.count('-')
    features['slash_count'] = url.count('/')
    
    # 2. 보안 및 신뢰성 관련
    features['is_https'] = 1 if url.startswith('https') else 0
    
    # 3. 비정상적인 패턴 (숫자 비율)
    digits = re.findall(r'\d', url)
    features['digit_ratio'] = len(digits) / len(url) if len(url) > 0 else 0
    
    # 4. 피싱 의심 키워드 포함 여부 (예시)
    suspicious_words = ['login', 'verify', 'bank', 'update', 'free', 'account']
    features['keyword_match'] = 1 if any(word in url.lower() for word in suspicious_words) else 0
    
    return features

# 전체 데이터프레임에 적용
print("🧠 특징 추출 중...")
features_list = total_df['url'].apply(extract_features).tolist()
X = pd.DataFrame(features_list)  # 학습 데이터 (Feature)
y = total_df['label']           # 정답 (Label: 0 또는 1)

print("✅ 특징 추출 완료!")
print(X.head())









from sklearn.model_selection import train_test_split

# 학습용 80%, 테스트용 20%로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"학습 데이터 개수: {len(X_train)}")
print(f"테스트 데이터 개수: {len(X_test)}")









from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 모델 생성 및 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측 및 성능 확인
y_pred = model.predict(X_test)

print("\n" + "="*20 + " 모델 평가 결과 " + "="*20)
print(f"정확도(Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("\n[상세 보고서]")
print(classification_report(y_test, y_pred))









import joblib

# 모델 저장: protocol=4, compress=0 권장(3.12 서버에서 3.13 pickle opcode 이슈 완화)
try:
    joblib.dump(model, "gnn_model.pkl", compress=0, protocol=4)
except TypeError:
    try:
        joblib.dump(model, "gnn_model.pkl", protocol=4)
    except TypeError:
        joblib.dump(model, "gnn_model.pkl")
try:
    joblib.dump(X.columns.tolist(), "gnn_model_features.pkl", compress=0, protocol=4)
except TypeError:
    try:
        joblib.dump(X.columns.tolist(), "gnn_model_features.pkl", protocol=4)
    except TypeError:
        joblib.dump(X.columns.tolist(), "gnn_model_features.pkl")

print("✅ 모델 저장 완료: gnn_model.pkl")









import torch
from torch_geometric.data import Data

# 각 노드(URL)의 특징값 (아까 추출한 수치 데이터)
# x = [노드 개수, 특징 개수]
x = torch.tensor([[10, 2, 0.1], [15, 3, 0.2], [12, 1, 0.05]], dtype=torch.float)

# 연결 관계 (0번 URL이 1번으로 리다이렉트됨, 1번이 2번으로...)
# edge_index = [2, 연결 선 개수]
edge_index = torch.tensor([[0, 1],
                           [1, 2]], dtype=torch.long)

# 정답 (0: 안전, 1: 악성)
y = torch.tensor([0, 1, 1], dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)









import torch

# 노드 개수 확인
num_nodes = data.x.size(0)

# 1. 모든 마스크를 False로 초기화
data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

# 2. 랜덤하게 인덱스 섞기
indices = torch.randperm(num_nodes)
train_size = int(num_nodes * 0.8) # 80%를 학습용으로

# 3. 앞쪽 80%는 학습용, 뒤쪽 20%는 테스트용으로 설정
data.train_mask[indices[:train_size]] = True
data.test_mask[indices[train_size:]] = True

print(f"📊 학습 데이터: {data.train_mask.sum()}개")
print(f"📊 테스트 데이터: {data.test_mask.sum()}개")









def test():
    model.eval()
    logits, accs = model(data.to(device)), []
    for mask in [data.train_mask, data.test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

train_acc, test_acc = test()
print(f'최종 학습 정확도: {train_acc:.4f}, 테스트 정확도: {test_acc:.4f}')









from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_models(y_true, y_pred_single, y_pred_ensemble):
    # 단일 모델 vs 통합 모델 오탐 비교
    cm_single = confusion_matrix(y_true, y_pred_single)
    cm_ensemble = confusion_matrix(y_true, y_pred_ensemble)
    
    print("✅ 단일 모델 오탐(FP):", cm_single[0][1])
    print("🚀 통합 모델 오탐(FP):", cm_ensemble[0][1])
    
    # 오탐이 얼마나 줄었는지 확인하는 게 목표!
    if cm_ensemble[0][1] < cm_single[0][1]:
        print(f"결과: 오탐이 {cm_single[0][1] - cm_ensemble[0][1]}건 감소했습니다!")
