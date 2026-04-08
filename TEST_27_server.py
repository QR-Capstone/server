import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer
import requests
from curl_cffi import requests as curl_requests
from bs4 import BeautifulSoup
import re
import os
from urllib.parse import unquote
import time
import io
from contextlib import redirect_stdout
from playwright.sync_api import sync_playwright

# ==========================================
# 🌟 [설정] 모델 파일 경로 설정
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_FILE = os.path.join(BASE_DIR, 'kobert_phishing_model_weights.pt')

# 전역 변수 선언
device = None
tokenizer = None
model = None
playwright_manager = None
engine_initialized = False

# ==========================================
# 🔥 Playwright 관리 클래스 (Singleton)
# ==========================================
class PlaywrightManager:
    def __init__(self):
        print("\n  [시스템] Playwright 엔진 백그라운드 기동 중...")
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=True,
            args=["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"]
        )
        self.context = self.browser.new_context()
        print("  [시스템] [OK] 브라우저 엔진 준비 완료.")

    def get_page(self):
        page = self.context.new_page()
        def intercept_route(route):
            if route.request.resource_type in ["image", "media"]:
                route.abort()
            else:
                route.continue_()
        page.route("**/*", intercept_route)
        return page

    def close(self):
        self.context.close()
        self.browser.close()
        self.playwright.stop()

# ==========================================
# 🛠️ 텍스트 전처리 및 수집 함수 (기존 100% 동일)
# ==========================================
_RE_PHONE = re.compile(r"\b\d{2,3}[-\s]?\d{3,4}[-\s]?\d{4}\b")
_RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

def redact_pii(text: str) -> str:
    text = _RE_PHONE.sub("[전화번호]", text)
    text = _RE_EMAIL.sub("[이메일]", text)
    text = re.sub(r"\b[\d\s]*\*{2,}[\d\s]*\b", "[카드번호_형태]", text)
    text = re.sub(r"\b\d{9,}\b", "[장문숫자]", text)
    return text

def extract_with_html_ultimate_clean(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for noise in soup(["script", "style", "noscript", "iframe"]):
        noise.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    type_map = { "tel": "전화번호", "email": "이메일", "password": "비밀번호", "text": "텍스트", "number": "숫자", "checkbox": "체크박스" }
    sensitive_map = { "account": "계좌번호", "acc_no": "계좌번호", "bank": "계좌번호", "resident": "주민등록번호", "jumin": "주민등록번호", "rrn": "주민등록번호", "card_num": "카드번호", "card_no": "카드번호", "cc_num": "카드번호", "cvc": "카드보안코드", "cvv": "카드보안코드" }
    raw_inputs = []
    for input_tag in soup.find_all("input"):
        i_type = (input_tag.get("type", "text") or "text").lower()
        if i_type in ["hidden", "submit", "button", "image"]: continue
        attr_context = f"{input_tag.get('name','')}_{input_tag.get('id','')}_{input_tag.get('placeholder','')}".lower()
        found_sensitive = False
        for eng_key, kor_val in sensitive_map.items():
            if eng_key in attr_context:
                raw_inputs.append(kor_val); found_sensitive = True; break
        if not found_sensitive: raw_inputs.append(type_map.get(i_type, i_type))
    inputs_str = ", ".join(list(dict.fromkeys(raw_inputs)))
    raw_buttons = []
    ignore_keys = ["bksp", "shift", "enter", "lang", "space", "caps", "tab", "clear", "done", "search"]
    for btn in soup.find_all(["button"]):
        btn_text = btn.get_text(strip=True)
        if btn_text and 1 < len(btn_text) <= 20 and btn_text.lower() not in ignore_keys: raw_buttons.append(btn_text)
    for inp_btn in soup.find_all("input", attrs={"type": ["submit", "button"]}):
        btn_val = (inp_btn.get("value", "") or "").strip()
        if btn_val and 1 < len(btn_val) <= 20 and btn_val.lower() not in ignore_keys: raw_buttons.append(btn_val)
    buttons_str = ", ".join(list(dict.fromkeys(raw_buttons)))
    blacklist_words = ["바로가기", "레이어", "새창", "건너뛰기", "닫기", "펼치기", "열기", "선택됨", "최근 검색어", "자동완성", "전체삭제", "도움말"]
    number_pattern = re.compile(r"^[\d,\.%+\-\s]+$")
    extracted_texts = []
    short_text_count = 0
    for text in soup.stripped_strings:
        if any(bad_word in text for bad_word in blacklist_words): continue
        if number_pattern.match(text): continue
        text_len = len(text)
        if 2 < text_len <= 25:
            if short_text_count < 15: extracted_texts.append(text); short_text_count += 1
        elif 25 < text_len <= 500: extracted_texts.append(text)
    unique_texts = []
    seen = set()
    for t in extracted_texts:
        if t not in seen: seen.add(t); unique_texts.append(t)
    main_text = " ".join(unique_texts)[:300]

    context_sentences = []
    if title: context_sentences.append(f"이 웹페이지의 제목은 '{title}'입니다.")
    if main_text: context_sentences.append(f"화면에 표시된 주요 안내 사항은 다음과 같습니다. {main_text}")
    if inputs_str and buttons_str:
        if any(k in inputs_str for k in ["계좌", "주민", "카드"]): context_sentences.append(f"이 페이지는 보안이 필요한 '{inputs_str}' 입력을 요구하며, '{buttons_str}' 버튼이 존재합니다.")
        else: context_sentences.append(f"이 페이지는 사용자에게 '{inputs_str}' 입력을 요청하며, '{buttons_str}' 버튼이 존재합니다.")
    elif inputs_str: context_sentences.append(f"이 페이지는 사용자에게 '{inputs_str}' 입력을 요청합니다.")
    elif buttons_str: context_sentences.append(f"이 페이지에는 '{buttons_str}' 버튼이 존재합니다.")

    final_text = " ".join(context_sentences)
    final_text = re.sub(r"\s+", " ", final_text).strip()[:1000]
    return redact_pii(final_text)

def extract_with_requests(url: str) -> str:
    html = ""
    if url.startswith("file://") or url[1:3] == ":\\":
        try:
            file_path = unquote(url.replace("file:///", "").replace("file://", ""))
            with open(file_path, "rb") as f: raw = f.read()
            html = raw.decode("utf-8", errors="ignore")
        except Exception as e: return f"[오류] 로컬 파일을 읽을 수 없습니다: {e}"
    else:
        try:
            timeout = int(os.getenv("KOBERT_HTTP_TIMEOUT", "5"))
            response = curl_requests.get(url, impersonate="chrome116", timeout=timeout)
            html = response.content.decode('utf-8', errors='replace')
        except Exception as e: return f"[오류] 네트워크 접속 문제: {e}"
    return extract_with_html_ultimate_clean(html)

def extract_with_playwright(url: str, is_warmup=False) -> str:
    if not is_warmup: print("  [알림] 2단계: Playwright 정밀 스캔을 시작합니다.")
    page = None
    try:
        global playwright_manager
        if playwright_manager is None:
            playwright_manager = PlaywrightManager()
        page = playwright_manager.get_page()
        try: page.goto(url, timeout=8000, wait_until="networkidle")
        except: pass
        if not is_warmup:
            wait_time = 0
            max_wait_seconds = float(os.getenv("KOBERT_PW_MAX_WAIT_SECONDS", "1"))
            while wait_time < max_wait_seconds:
                current_html = page.content()
                soup_test = BeautifulSoup(current_html, "html.parser")
                if len(soup_test.get_text(strip=True)) > 150: break
                page.wait_for_timeout(1000); wait_time += 1
        full_html = page.content()
        return extract_with_html_ultimate_clean(full_html)
    finally:
        if page: page.close()


# ==========================================
# 🚀 1. 조원의 서버에서 호출하는 [예열 함수]
# ==========================================
def warmup_engine(include_pw=True):
    """서버 기동 시 조원의 코드에서 호출되는 엔진 초기화 및 예열 함수입니다."""
    global device, tokenizer, model, playwright_manager, engine_initialized

    if engine_initialized:
        return {"status": "already_initialized"}

    print("  [엔진] 모델 로딩을 시작합니다...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
    model = BertForSequenceClassification.from_pretrained('monologg/kobert', num_labels=2)

    if os.path.exists(WEIGHTS_FILE):
        model.load_state_dict(
            torch.load(WEIGHTS_FILE, map_location=device, weights_only=False)
        )
        model.to(device)
        model.eval()
    else:
        raise FileNotFoundError(f"모델 가중치 파일을 찾을 수 없습니다: {WEIGHTS_FILE}")

    # 가짜 텍스트로 모델 예열
    dummy_inputs = tokenizer("예열 테스트", return_tensors="pt", max_length=128, padding='max_length', truncation=True)
    with torch.no_grad():
        _ = model(dummy_inputs['input_ids'].to(device), attention_mask=dummy_inputs['attention_mask'].to(device))

    # Playwright 예열 여부 (조원 서버 설정에 따름)
    pw_status = "skipped"
    if include_pw:
        try:
            playwright_manager = PlaywrightManager()
            try:
                extract_with_playwright("about:blank", is_warmup=True)
                pw_status = "warmed_up"
            except Exception as e:
                pw_status = f"error: {e}"
        except Exception as e:
            pw_status = f"init_error: {e}"

    engine_initialized = True
    print("  [엔진] [OK] 초기화 및 예열 완료!")

    return {
        "model_loaded": True,
        "device": str(device),
        "playwright_status": pw_status
    }


# ==========================================
# 🚀 2. 조원의 서버에서 호출하는 [검증 및 결과 반환 함수]
# ==========================================
def predict_phishing_result(target_url):
    global device, tokenizer, model, engine_initialized

    if not engine_initialized:
        raise RuntimeError("엔진이 아직 예열/초기화되지 않았습니다.")

    if not (target_url.startswith("http") or ":" in target_url or target_url.startswith("/")):
        target_url = "https://" + target_url

    f = io.StringIO()
    
    with redirect_stdout(f):
        print(f"\n[{target_url}] 데이터 추출 시작...")
        processed_text = extract_with_requests(target_url)

        if len(processed_text) < 150 or processed_text.startswith("[오류]"):
            use_pw = os.getenv("USE_PLAYWRIGHT_IN_ANALYZE", "0") == "1"
            if use_pw:
                print("  [!] 정밀 스캔으로 전환합니다.")
                try:
                    processed_text = extract_with_playwright(target_url, is_warmup=False)
                except Exception as e:
                    processed_text = f"[오류] Playwright 스캔 실패: {e}"
            else:
                print("  [!] 정밀 스캔(Playwright) 스킵. USE_PLAYWRIGHT_IN_ANALYZE=0")

        # 🚨 [수정 1] 판별 불가 상태일 때 딱 한 줄만 반환!
        if processed_text.startswith("[오류]") or processed_text.startswith("[판별 보류]"):
            return {
                "judgment": "unknown",
                "riskLevel": "UNKNOWN",
                "risklevel": "UNKNOWN",
            }

        max_len = int(os.getenv("KOBERT_MAX_LEN", "512"))
        inputs = tokenizer(
            processed_text,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        input_ids, attention_mask = inputs['input_ids'].to(device), inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1)[0]

        prob_normal = probs[0].item() * 100
        prob_phishing = probs[1].item() * 100
        is_phishing = prob_phishing > 50

        if is_phishing:
            final_judgment = "unnormal"
        else:
            final_judgment = "normal"

    # 🚨 [수정 2] 정상 or 피싱 상태일 때도 딱 한 줄만 반환!
    risk_level = "HIGH" if final_judgment == "unnormal" else "LOW"
    return {
        "judgment": final_judgment,
        "riskLevel": risk_level,
        "risklevel": risk_level,
    }