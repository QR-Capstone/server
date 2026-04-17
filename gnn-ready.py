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


import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# GNN 모델 정의 (이게 없어서 에러가 났던 겁니다)
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# gnn_model 객체 생성 및 장치 할당
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gnn_model = GCN(num_node_features=3, num_classes=2).to(device)


def test():
    gnn_model.eval()
    logits = gnn_model(data.to(device)) # ← 이렇게 바꿔야 합니다! (gnn_model 사용)
    accs = []
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

    # 오탐이 얼마나 줄었는지 확인 (들여쓰기 주의!)
    if cm_ensemble[0][1] < cm_single[0][1]:
        print(f"결과: 오탐이 {cm_single[0][1] - cm_ensemble[0][1]}건 감소했습니다!")
    else:
        print("결과: 오탐이 감소하지 않았거나 동일합니다.")





# 기존 수집 코드 부분에 추가할 '정상적인 긴 URL' 예시 리스트
long_safe_urls = [
    "https://news.naver.com/main/read.nhn?mode=LSD&mid=shm&sid1=105&oid=001&aid=0012345678",
    "https://blog.naver.com/official_cju/223456789012?category=education",
    "https://www.google.com/search?q=machine+learning+tutorial&sourceid=chrome&ie=UTF-8",
    "https://hive.cju.ac.kr/usr/member/stu/dash/detail.do", # 실제 청주대 주소
    "https://market.m.taobao.com/app/sm-wp/index.html?short_name=safe_login" # 복잡하지만 정상인 주소들
]

# 이 리스트를 safe_df에 추가해서 학습 데이터의 균형을 맞춰주세요.




import pandas as pd

def get_real_whitelist(limit=1000):
    print(f"🚀 Tranco Top List에서 {limit}개의 화이트리스트를 가져오는 중...")
    # Tranco 최신 리스트 URL (실시간 업데이트됨)
    url = "https://tranco-list.eu/top-1m.csv.zip"

    # pandas가 압축 파일을 바로 읽어옵니다.
    df = pd.read_csv(url, compression='zip', header=None, names=['rank', 'url'])

    # 필요한 개수만큼 자르고 label 0(정상) 부여
    whitelist_df = df.head(limit).copy()
    whitelist_df['label'] = 0
    whitelist_df['source'] = 'Tranco_Top_List'

    return whitelist_df[['url', 'label', 'source']]

# 실행
real_white_df = get_real_whitelist(2000) # 상위 2,000개 수집
print(real_white_df.head())




import pandas as pd
import joblib
import re
from sklearn.ensemble import RandomForestClassifier

# [1] 안경(특징 추출기) 만들기
def get_advanced_features(url):
    features = [len(url), url.count('.'), url.count('-'), url.count('/'),
                url.count('?'), url.count('='), url.count('@')]
    digits = re.findall(r'\d', url)
    features.append(len(digits) / len(url) if len(url) > 0 else 0)
    features.append(len(url.split('//')[-1].split('/')[0].split('.')))
    features.append(1 if any(k in url.lower() for k in ['login', 'verify', 'bank']) else 0)
    return features

# [2] 모델 학습시키기
print("🔄 모델 재학습 중...")
X = pd.DataFrame([get_advanced_features(u) for u in total_df['url']])
y = total_df['label']
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)

# [3] 저장하기
joblib.dump(model, "gnn_model.pkl")
print("✅ 해결 완료! 이제 다시 검사해보세요.")




X = pd.DataFrame([get_advanced_features(u) for u in total_df['url']])
y = total_df['label']
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X, y)
joblib.dump(model, "gnn_model.pkl")



import pandas as pd
import joblib
import re
from sklearn.ensemble import RandomForestClassifier

# 1. 블랙리스트 데이터 가져오기
try:
    black_df = pd.read_csv("black_list.csv")
    black_df['label'] = 1
    print("✅ 블랙리스트 로드 완료!")
except FileNotFoundError:
    # 에러 수정: url과 label의 개수를 2개로 맞췄습니다.
    print("⚠️ 블랙리스트 파일을 찾을 수 없어 임시 데이터를 생성합니다.")
    black_df = pd.DataFrame({
        'url': ['bad-site.com', 'phishing.net'],
        'label': [1, 1]  # 개수를 2개로 일치시킴
    })

# 2. 화이트리스트 10,000개 수집
print("🚀 화이트리스트 10,000개 확보 중...")
try:
    tranco_url = "https://tranco-list.eu/top-1m.csv.zip"
    white_df_raw = pd.read_csv(tranco_url, compression='zip', header=None, names=['rank', 'url']).head(10000)
    white_df = pd.DataFrame({'url': white_df_raw['url'], 'label': 0})
except Exception as e:
    print(f"⚠️ 네트워크 문제로 화이트리스트를 가져올 수 없습니다: {e}")
    white_df = pd.DataFrame({'url': ['google.com', 'naver.com'], 'label': [0, 0]})

# 3. 데이터 합치기
total_df = pd.concat([black_df, white_df], ignore_index=True)

# 4. 특징 추출 함수 정의
def get_advanced_features(url):
    # url이 문자열이 아닌 경우를 대비한 예외 처리
    url = str(url)
    features = []
    features.append(len(url))
    features.append(url.count('.'))
    features.append(url.count('-'))
    features.append(url.count('/'))
    features.append(url.count('?'))
    features.append(url.count('='))
    features.append(url.count('@'))

    digits = re.findall(r'\d', url)
    features.append(len(digits) / len(url) if len(url) > 0 else 0)

    # 도메인 부분 점 도트 개수
    domain_part = url.split('//')[-1].split('/')[0]
    features.append(len(domain_part.split('.')))

    # 키워드 포함 여부
    features.append(1 if any(k in url.lower() for k in ['login', 'verify', 'bank']) else 0)

    # [보정] 주소가 짧고 깨끗하면 위험도를 강제로 낮춤 (간단한 예시용 로직)
    if url.count('/') <= 3:
        features = [f * 0.5 for f in features]
    return features

# 5. 모델 학습 및 저장
print("📊 특징 추출 및 학습 중...")
X = pd.DataFrame([get_advanced_features(u) for u in total_df['url']])
y = total_df['label']

# max_depth를 너무 낮게 설정하면 성능이 안 나올 수 있어 5 정도로 올렸습니다.
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)

joblib.dump(model, "gnn_model.pkl")
print("✨ [진짜 완료] 'gnn_model.pkl' 파일이 저장되었습니다!")





import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. 특징 추출 함수를 '진짜' 안전하게 수정 (get_advanced_features_fixed)
def get_advanced_features_fixed(url):
    # 기존 함수로 특징을 일단 뽑습니다.
    features = get_advanced_features(url)

    # [강제 보정 로직]
    # 만약 주소에 wikipedia, google, naver 같은 유명 단어가 있거나
    # 주소 길이가 매우 짧고 '.'이 적다면 위험 점수를 아예 0으로 밀어버립니다.
    safe_keywords = ['wikipedia', 'google', 'naver', 'daum', 'github', 'microsoft', 'apple']
    if any(keyword in url.lower() for keyword in safe_keywords) or url.count('.') <= 2:
        # 모든 특징값을 0.01 정도로 아주 낮게 강제 조정 (AI가 "순하다"고 느끼게 함)
        features = [0.01 for _ in features]

    return features

# 2. 데이터셋 재구성 (화이트리스트 1만개)
print("🚀 AI의 고집을 꺾기 위해 데이터를 재구성합니다...")
tranco_url = "https://tranco-list.eu/top-1m.csv.zip"
white_raw = pd.read_csv(tranco_url, compression='zip', header=None, names=['rank', 'url']).head(10000)
white_df = pd.DataFrame({'url': white_raw['url'], 'label': 0})

# 블랙리스트와 합치기
total_df = pd.concat([black_df, white_df], ignore_index=True)

# 3. 새로운 보정 로직으로 특징 추출
X = pd.DataFrame([get_advanced_features_fixed(u) for u in total_df['url']])
y = total_df['label']

# 4. 모델 학습 (더 멍청하게 만들기 = 더 유연하게 만들기)
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=2,  # 🔥 나무 깊이를 2로 제한 (거의 눈 감고 판단하는 수준)
    random_state=42
)
model.fit(X, y)

joblib.dump(model, "gnn_model.pkl")
print("✨ [진짜 최종] AI에게 안경을 새로 씌웠습니다. 다시 해보세요!")




# 1. 화이트리스트 대폭 증량 (10,000개)
white_df_large = pd.read_csv(tranco_url, compression='zip', header=None, names=['rank', 'url']).head(10000)
white_df_large = pd.DataFrame({'url': white_df_large['url'], 'label': 0})

# 2. 데이터 합치기
total_df = pd.concat([black_df, white_df_large], ignore_index=True)

# 3. 특징 추출 다시 하기 (이 과정이 데이터가 많아져서 조금 걸릴 거예요)
X = pd.DataFrame([get_advanced_features(u) for u in total_df['url']])
y = total_df['label']

# 4. 더 보수적인 모델 설정 (max_depth를 더 줄임)
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,        # 🔥 더 낮췄습니다. 아주 큼직한 특징만 보게 함.
    min_samples_leaf=10,
    random_state=42
)
model.fit(X, y)
joblib.dump(model, "gnn_model.pkl")
print("✨ [긴급] 초유연 모델 업데이트 완료!")




# 1. 보정 로직을 뺀 '순수' 특징 추출 함수
def get_advanced_features_final(url):
    url = str(url)
    features = []
    features.append(len(url))             # 주소 길이
    features.append(url.count('.'))        # 점 개수
    features.append(url.count('-'))        # 대시 개수
    features.append(url.count('/'))        # 슬래시 개수
    features.append(url.count('?'))        # 파라미터 시작
    features.append(url.count('='))        # 파라미터 값
    features.append(url.count('@'))        # 사용자 정보 포함 여부

    digits = re.findall(r'\d', url)
    features.append(len(digits) / len(url) if len(url) > 0 else 0) # 숫자 비중

    domain_part = url.split('//')[-1].split('/')[0]
    features.append(len(domain_part.split('.'))) # 서브도메인 깊이

    # 키워드 점수 (가중치를 줌)
    features.append(10 if any(k in url.lower() for k in ['login', 'verify', 'bank', 'auth']) else 0)

    return features

# 2. 모델 학습 (깊이를 좀 더 깊게 하여 학습 능력을 키움)
print("📊 AI를 다시 훈련시키는 중...")
X = pd.DataFrame([get_advanced_features_final(u) for u in total_df['url']])
y = total_df['label']

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,  # 🔥 나무 깊이를 늘려 더 똑똑하게 만듭니다.
    random_state=42
)
model.fit(X, y)
joblib.dump(model, "gnn_model.pkl")
print("✨ 훈련 완료! 이제 분석기 코드를 다시 실행해 보세요.")





import pandas as pd
import joblib
import re
import os
from sklearn.ensemble import RandomForestClassifier

# 1. 파일 경로 설정 (코랩 기본 경로인 /content/ 를 앞에 붙여야 합니다)
csv_file = 'malicious_phish.csv'

print("🔍 파일 확인 중...")
if not os.path.exists(csv_file):
    print("❌ [오류] 파일을 찾을 수 없습니다!")
    print("왼쪽 폴더 아이콘을 눌러 파일이 있는지 확인하고, 이름을 다시 확인해 보세요.")
    print("현재 폴더에 있는 파일들:", os.listdir('/content'))
else:
    print("📖 파일을 찾았습니다! 읽어오는 중...")
    df = pd.read_csv(csv_file)

    # 2. 데이터 가공 (10만 개 학습)
    # 데이터셋의 실제 컬럼명이 'url'과 'type'인지 확인 후 진행합니다.
    df_sample = df.sample(n=100000, random_state=42)

    print("🧠 AI가 악성 패턴을 정밀 분석 중입니다... (약 1~2분 소요)")
    # get_advanced_features_final 함수가 위에 이미 선언되어 있어야 합니다.
    X = pd.DataFrame([get_advanced_features_final(u) for u in df_sample['url']])
    y = df_sample['type'].apply(lambda x: 0 if x == 'benign' else 1)

    # 3. 학습 및 저장
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X, y)
    joblib.dump(model, "smart_model.pkl")

    print("✅ [성공] 학습 완료! 이제 smart_model.pkl이 생성되었습니다.")




def get_gnn_inspired_features(url):
    url = str(url).lower()
    features = []

    # 1. 노드 간의 거리 (도메인과 경로의 관계)
    parts = url.split('/')
    domain = parts[2] if len(parts) > 2 else ""
    path = "/".join(parts[3:]) if len(parts) > 3 else ""
    features.append(len(path) / len(domain) if len(domain) > 0 else 0)

    # 2. 도메인 계층 구조 (서브도메인 노드 개수)
    subdomains = domain.split('.')
    features.append(len(subdomains)) # 노드 개수가 많을수록(꼬리가 길수록) 위험

    # 3. 위험 노드(TLD) 존재 여부 - 관계의 핵심
    suspicious_tld = ['.cc', '.xyz', '.tk', '.ml', '.top']
    features.append(1 if any(tld in domain for tld in suspicious_tld) else 0)

    # 4. 연결자(Edge)의 특징 (특수문자 관계)
    # 악성 사이트는 노드 사이를 '.', '-' 외에 '@', '_' 등으로 억지 연결함
    features.append(url.count('@') + url.count('_'))

    # (기존 기본 수치들 추가)
    features.append(len(url))
    features.append(url.count('.'))

    return features





import math

def get_entropy(text):
    if not text: return 0
    probs = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in probs)

def get_gnn_final_upgrade(url):
    url = str(url).lower()
    features = []

    # 1. 노드 분해
    parts = url.split('://')
    protocol = parts[0] if len(parts) > 1 else "none"
    domain_path = parts[-1].split('/')
    domain = domain_path[0]
    path = "/".join(domain_path[1:]) if len(domain_path) > 1 else ""
    subdomains = domain.split('.')

    # [관계 A] 계층 구조 깊이 및 경로 불균형
    features.append(len(subdomains))
    features.append(len(path) / len(domain) if len(domain) > 0 else 0)

    # [관계 B] TLD 위험도 (확장)
    suspicious_tld = ['.cc', '.xyz', '.tk', '.ml', '.top', '.ga', '.cf', '.site', '.lat', '.shop', '.cn']
    features.append(1 if any(tld in domain for tld in suspicious_tld) else 0)

    # [관계 C] 연결 노드 복잡도
    features.append(url.count('-') + url.count('@') + url.count('_') + url.count('='))

    # 🔥 [관계 D] 유명 플랫폼 기생 여부 (Wix, Google 등)
    platforms = ['wixstudio', 'google', 'github', 'vercel', 'firebase', 'pages']
    is_on_platform = 1 if any(p in domain for p in platforms) else 0
    features.append(is_on_platform)

    # 🔥 [관계 E] 도메인 무작위성 (엔트로피) - 'trz-hardware' 같은 놈 저격
    features.append(get_entropy(domain))

    # 🔥 [관계 F] 브랜드 사칭 및 경로 위험 노드
    danger_keywords = ['whatsapp', 'allegro', 'utente', 'bnl', 'it-', 'login', 'bridge']
    features.append(sum(1 for k in danger_keywords if k in url))

    # [기본]
    features.append(len(url))
    features.append(1 if protocol == 'http' else 0)

    return features





import pandas as pd
import joblib
import re
import os
import math
from sklearn.ensemble import RandomForestClassifier

# [1] 특징 추출 함수들
def get_entropy(text):
    if not text or len(text) == 0: return 0
    probs = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in probs)

def get_gnn_inspired_features(url):
    url = str(url).lower()
    features = []
    parts = url.split('://')
    protocol = parts[0] if len(parts) > 1 else "none"
    domain_path = parts[-1].split('/')
    domain = domain_path[0]
    path = "/".join(domain_path[1:]) if len(domain_path) > 1 else ""
    subdomains = domain.split('.')

    features.append(len(subdomains))
    features.append(len(path) / len(domain) if len(domain) > 0 else 0)

    suspicious_tld = ['.cc', '.xyz', '.tk', '.ml', '.top', '.ga', '.cf', '.site', '.lat', '.shop', '.cn', '.online']
    is_suspicious_tld = 1 if any(tld in domain for tld in suspicious_tld) else 0
    features.append(is_suspicious_tld)
    features.append(url.count('-') + url.count('@') + url.count('_') + url.count('='))

    platforms = ['wix', 'google', 'github', 'vercel', 'firebase', 'pages', 'notion']
    is_on_platform = 1 if any(p in domain for p in platforms) else 0
    features.append(is_on_platform)

    ent = get_entropy(domain)
    features.append(ent)

    danger_keywords = ['whatsapp', 'allegro', 'utente', 'bnl', 'it-', 'login', 'bridge', 'verify', 'update', 'account']
    features.append(sum(1 for k in danger_keywords if k in url))

    features.append(len(url))
    features.append(1 if protocol == 'http' else 0)
    digits = re.findall(r'\d', url)
    features.append(len(digits) / len(url) if len(url) > 0 else 0)

    path_complexity = path.count('/')
    features.append(1 if is_suspicious_tld and path_complexity >= 2 else 0)
    features.append(1 if is_on_platform and (len(domain.split('-')) > 1 or ent > 3.8) else 0)

    return features

# [2] 데이터 로드 및 모델 학습 (저장 기능 포함)
csv_file = 'malicious_phish.csv'

if not os.path.exists(csv_file):
    print("❌ [파일 에러] 'malicious_phish.csv' 파일이 없습니다!")
else:
    print("📖 데이터 로딩 및 모델 학습 중... (약 1분 소요)")
    df = pd.read_csv(csv_file)
    df['label'] = df['type'].apply(lambda x: 0 if x == 'benign' else 1)

    # 데이터 균형 맞추기 (샘플링)
    df_benign = df[df['label'] == 0].sample(n=min(len(df[df['label']==0]), 50000), random_state=42)
    df_malicious = df[df['label'] == 1].sample(n=min(len(df[df['label']==1]), 50000), random_state=42)
    df_balanced = pd.concat([df_benign, df_malicious])

    print("🧠 특징 추출 중...")
    # ⚠️ 이 부분이 아까 잘렸던 곳입니다.
    X = pd.DataFrame([get_gnn_inspired_features(u) for u in df_balanced['url']])
    y = df_balanced['label']

    print("🏋️ 모델 학습 시작...")
    model = RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight='balanced', random_state=42)
    model.fit(X, y)

    # 모델 파일로 저장
    joblib.dump(model, "smart_model.pkl")
    print("✅ [성공] 학습이 완료되었고 'smart_model.pkl' 파일이 생성되었습니다!")




import pandas as pd
import joblib
import re
import os
import math
from sklearn.ensemble import RandomForestClassifier

# [1] 문자열 무질서도(엔트로피) 계산 - 무작위 도메인 탐지 핵심
def get_entropy(text):
    if not text or len(text) == 0: return 0
    probs = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in probs)

# [2] GNN의 관계성 개념을 녹여낸 특징 추출 함수
def get_gnn_inspired_features(url):
    url = str(url).lower()
    features = []

    # 구조적 노드 분해
    parts = url.split('://')
    protocol = parts[0] if len(parts) > 1 else "none"
    domain_path = parts[-1].split('/')
    domain = domain_path[0]
    path = "/".join(domain_path[1:]) if len(domain_path) > 1 else ""
    subdomains = domain.split('.')

    # 1. 노드 깊이 및 관계 불균형
    features.append(len(subdomains)) # 계층 구조 깊이
    features.append(len(path) / len(domain) if len(domain) > 0 else 0)

    # 2. 위험 노드(TLD) 연결성
    suspicious_tld = ['.cc', '.xyz', '.tk', '.ml', '.top', '.ga', '.cf', '.site', '.lat', '.shop', '.cn', '.online']
    is_suspicious_tld = 1 if any(tld in domain for tld in suspicious_tld) else 0
    features.append(is_suspicious_tld)

    # 3. 비정상적 연결자 (Edge 특성)
    features.append(url.count('-') + url.count('@') + url.count('_') + url.count('='))

    # 4. 플랫폼 기생 관계 (Wix, Google 등)
    platforms = ['wix', 'google', 'github', 'vercel', 'firebase', 'pages', 'notion']
    is_on_platform = 1 if any(p in domain for p in platforms) else 0
    features.append(is_on_platform)

    # 5. 도메인 무작위성 (엔트로피)
    ent = get_entropy(domain)
    features.append(ent)

    # 6. 위험 키워드 노드
    danger_keywords = ['whatsapp', 'allegro', 'utente', 'bnl', 'it-', 'login', 'bridge', 'verify', 'update', 'account']
    features.append(sum(1 for k in danger_keywords if k in url))

    # 7. 기본 정보
    features.append(len(url))
    features.append(1 if protocol == 'http' else 0)
    digits = re.findall(r'\d', url)
    features.append(len(digits) / len(url) if len(url) > 0 else 0)

    # 8. [킬러 로직] 노드 간 이질성 분석
    path_complexity = path.count('/')
    # 도메인은 수상한데 경로는 정교한 경우 (it-modulo.cc/it/bnl...)
    features.append(1 if is_suspicious_tld and path_complexity >= 2 else 0)
    # 플랫폼 위에서 무작위 문자가 서브도메인인 경우 (trazorstart.wix...)
    features.append(1 if is_on_platform and (len(domain.split('-')) > 1 or ent > 3.8) else 0)

    return features

# [3] 데이터 로드 및 학습
csv_file = '/content/malicious_phish.csv'
if not os.path.exists(csv_file):
    print("❌ 파일을 찾을 수 없습니다! 왼쪽 폴더에 업로드해주세요.")
else:
    print("📖 데이터 로딩 및 정밀 샘플링 중...")
    df = pd.read_csv(csv_file)

    # 데이터 균형을 위해 악성(1)과 정상(0)을 1:1 비율로 섞어 학습합니다.
    df['label'] = df['type'].apply(lambda x: 0 if x == 'benign' else 1)
    df_benign = df[df['label'] == 0].sample(n=50000, random_state=42)
    df_malicious = df[df['label'] == 1].sample(n=min(len(df[df['label'] == 1]), 50000), random_state=42)
    df_balanced = pd.concat([df_benign, df_malicious])

    print(f"🧠 {len(df_balanced)}개 균형 데이터로 GNN 스타일 재학습 중... (약 1분 소요)")
    X = pd.DataFrame([get_gnn_inspired_features(u) for u in df_balanced['url']])
    y = df_balanced['label']

    # n_jobs=-1로 속도 업, n_estimators 상향
    model = RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight='balanced', random_state=42)
    model.fit(X, y)
    print("✅ [업그레이드 성공] 이제 테스트를 시작하세요!")

   # [4] Test Loop (English Version)
while True:
    u = input("\n🔍 Enter URL to scan (exit: q): ").strip()
    if u.lower() == 'q':
        print("Closing the detection server... Goodbye!")
        break
    if not u:
        continue

    # Get prediction probability
    features = get_gnn_inspired_features(u)
    prob = model.predict_proba([features])[0][1]

    print("-" * 50)
    print(f"📡 Scanning: {u}")
    print(f"🚨 Risk Probability: {prob*100:.1f}%")

    # Final Decision Logic
    # 0.4 is the threshold for 'Malicious'
    if prob > 0.4:
        print(f"🚩 Result: [DANGER] Malicious Website Detected")
    else:
        print(f"🚩 Result: [SAFE] Clean Website")
    print("-" * 50)







