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
import concurrent.futures

# 🔥 [비동기 및 쓰레딩 라이브러리]
import asyncio
from playwright.async_api import async_playwright
import threading
from urllib.parse import urlparse, urljoin

# ==========================================
# 🌟 [설정] 글로벌 변수 및 엔진 상태
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_FILE = os.path.join(BASE_DIR, 'kobert_phishing_model_weights.pt')

device = None
tokenizer = None
model = None
async_pw_manager = None    
engine_initialized = False

# ==========================================
# 🔥 [엔진 대통합] 영구 대기형 비동기 병렬 브라우저 풀
# ==========================================
class AsyncPlaywrightPool:
    def __init__(self):
        print("  [시스템] 초고속 비동기 마스터 브라우저 기동 중...")
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.thread.start()
        
        self.playwright = None
        self.browser = None
        
        future = asyncio.run_coroutine_threadsafe(self._init_browser(), self.loop)
        future.result() 

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _init_browser(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"]
        )
        print("  [시스템] [OK] 마스터 브라우저 풀 준비 완료. (1, 2-Depth 공용)")

    async def _fetch_single(self, url, timeout_ms=6000, wait_sec=3.5):
        context = await self.browser.new_context()
        page = await context.new_page()
        
        async def intercept_route(route):
            req_url = route.request.url.lower()
            if route.request.resource_type in ["image", "media", "font"] or \
               any(ad in req_url for ad in ["analytics", "tracker", "pixel", "adsystem", "doubleclick"]):
                await route.abort()
            else:
                await route.continue_()
        await page.route("**/*", intercept_route)

        try:
            try:
                await page.goto(url, timeout=timeout_ms, wait_until="commit")
            except Exception:
                pass 

            wait_time = 0
            while wait_time < wait_sec:
                current_html = await page.content()
                soup_test = BeautifulSoup(current_html, "html.parser")
                if len(soup_test.get_text(strip=True)) > 150: 
                    break
                await page.wait_for_timeout(250)
                wait_time += 0.25

            full_html = await page.content()
            clean_text = extract_with_html_ultimate_clean(full_html)
            return url, clean_text, full_html 
        except Exception as e:
            return url, f"[오류] {e}", ""
        finally:
            await context.close()

    async def _scrape_all(self, urls):
        tasks = [self._fetch_single(url) for url in urls]
        return await asyncio.gather(*tasks)

    def scrape_parallel(self, urls):
        future = asyncio.run_coroutine_threadsafe(self._scrape_all(urls), self.loop)
        return future.result()


# ==========================================
# 🛠️ 텍스트 전처리 및 수집 함수
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

def extract_with_requests_and_raw_html(url: str):
    html = ""
    if url.startswith("file://") or url[1:3] == ":\\":
        try:
            file_path = unquote(url.replace("file:///", "").replace("file://", ""))
            with open(file_path, "rb") as f: raw = f.read()
            try: html = raw.decode("utf-8")
            except UnicodeDecodeError: html = raw.decode("euc-kr", errors="ignore")
        except Exception: return "[오류] 로컬 파일을 읽을 수 없습니다", ""
    else:
        try:
            timeout = float(os.getenv("KOBERT_HTTP_TIMEOUT", "1.2"))
            response = curl_requests.get(url, impersonate="chrome116", timeout=timeout)
            raw_bytes = response.content
            try: html = raw_bytes.decode('utf-8')
            except UnicodeDecodeError:
                try: html = raw_bytes.decode('euc-kr')
                except UnicodeDecodeError: html = raw_bytes.decode('utf-8', errors='replace')
        except Exception:
            return "[오류] 네트워크 접속 문제 (Timeout)", ""
    return extract_with_html_ultimate_clean(html), html

def extract_with_playwright_and_raw_html(url, is_warmup=False):
    global async_pw_manager
    try:
        if not async_pw_manager:
            async_pw_manager = AsyncPlaywrightPool()
        
        results = async_pw_manager.scrape_parallel([url])
        if results:
            p_url, p_text, p_html = results[0]
            return p_text, p_html
    except Exception:
        pass
    return "[오류] Playwright 스캔 실패", ""

def extract_deep_links(raw_html, base_url, max_links=2): 
    soup = BeautifulSoup(raw_html, "html.parser")
    priority_links, normal_links = [], []
    target_keywords = ["로그인", "login", "회원가입", "가입", "sign in", "sign up", "본인인증", "인증", "비밀번호", "내정보"]
    
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        if href.startswith("#") or "javascript:" in href.lower() or href == "/": continue
        
        if not href.startswith("http"):
            full_url = urljoin(base_url, href)
        else: 
            full_url = href
            
        if full_url == base_url or full_url == base_url + "/": continue
            
        link_text = a_tag.get_text(strip=True).lower()
        href_lower = href.lower()
        
        is_priority = any(kw in link_text or kw in href_lower for kw in target_keywords)
                
        if is_priority:
            if full_url not in priority_links: priority_links.append(full_url)
        else:
            if full_url not in normal_links: normal_links.append(full_url)

    final_links = []
    for link in priority_links:
        if link not in final_links:
            final_links.append(link)
            if len(final_links) >= max_links: return final_links
    for link in normal_links:
        if link not in final_links:
            final_links.append(link)
            if len(final_links) >= max_links: return final_links
    return final_links

# ==========================================
# 🚀 본 서버 연동용 메인 판단 함수
# ==========================================
def warmup_engine(include_pw=True):
    global device, tokenizer, model, async_pw_manager, engine_initialized
    if engine_initialized: return {"status": "already_initialized"}
    
    print("\n  [시스템] AI 모델(KoBERT) 가중치 로딩 중...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
    model = BertForSequenceClassification.from_pretrained('monologg/kobert', num_labels=2)
    
    if os.path.exists(WEIGHTS_FILE):
        model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device, weights_only=False))
        model.to(device)
        model.eval()
    else: raise FileNotFoundError(f"모델 가중치 파일을 찾을 수 없습니다: {WEIGHTS_FILE}")

    dummy_inputs = tokenizer("예열 테스트", return_tensors="pt", max_length=128, padding='max_length', truncation=True)
    with torch.no_grad(): _ = model(dummy_inputs['input_ids'].to(device), attention_mask=dummy_inputs['attention_mask'].to(device))

    if include_pw:
        try:
            async_pw_manager = AsyncPlaywrightPool()
            async_pw_manager.scrape_parallel(["about:blank"]) 
        except Exception: pass

    engine_initialized = True
    print("  [시스템] [OK] 큐싱 디펜더 엔진 초기화 및 예열 완료!\n")
    return {"model_loaded": True, "device": str(device)}


def predict_phishing_result(target_url):
    global device, tokenizer, model, engine_initialized, async_pw_manager

    if not engine_initialized:
        try: warmup_engine(include_pw=True)
        except Exception as e: 
            return {"judgment": "unknown", "riskLevel": "UNKNOWN", "risklevel": "UNKNOWN", "error": f"엔진 예열 실패: {e}", "detectedUrl": target_url}

    if not (target_url.startswith("http") or ":" in target_url or target_url.startswith("/")): 
        target_url = "https://" + target_url

    safe_tlds = [".go.kr", ".ac.kr", ".edu", ".mil.kr", ".ms.kr"]

    safe_official_domains = [
        "nonghyup.com", "kbstar.com", "shinhan.com", "wooribank.com",
        "hanabank.com", "kakaobank.com", "tossbank.com", "kbanknow.com", "ibk.co.kr"
    ]


    try:
        domain = urlparse(target_url).netloc.lower()
        if any(domain.endswith(tld) for tld in safe_tlds) or \
           any(domain == d or domain.endswith("." + d) for d in safe_official_domains):
            return {"judgment": "normal", "riskLevel": "LOW", "risklevel": "LOW", "detectedUrl": target_url}
    except Exception:
        pass

    use_pw = os.getenv("USE_PLAYWRIGHT_IN_ANALYZE", "1") == "1"
    max_len = int(os.getenv("KOBERT_MAX_LEN", "512"))
    
    # 🔥 [공통 설정] 점수 보정을 위한 휴리스틱 키워드 사전
    high_risk_keywords = ["통신요금 담보", "신불자", "내구제", "폰테크", "신용등급 무관", "무직자 대출", "통신연체자", "비상장 주식", "공모주 청약", "원금 보장", "수익 보장", "투자 지원금", "리딩방"]
    action_keywords = ["비밀번호", "계좌", "로그인", "login", "주민번호", "주민등록번호", "인증번호", "입력을 요구"]

    # ----------------------------------------------------
    # 🌟 [1단계] 루트 URL 검사
    # ----------------------------------------------------
    processed_text, raw_html = extract_with_requests_and_raw_html(target_url)

    if "Suspected phishing site" in processed_text or "Cloudflare Ray ID" in processed_text:
        return {"judgment": "unnormal", "riskLevel": "HIGH", "risklevel": "HIGH", "detectedUrl": target_url}
    
    if len(processed_text) < 150 or processed_text.startswith("[오류]"):
        if use_pw:
            try: processed_text, raw_html = extract_with_playwright_and_raw_html(target_url, is_warmup=False)
            except Exception: pass

    if processed_text.startswith("[오류]") or processed_text.startswith("[판별 보류]"):
        return {"judgment": "unknown", "riskLevel": "UNKNOWN", "risklevel": "UNKNOWN", "detectedUrl": target_url}

    inputs = tokenizer(processed_text, max_length=max_len, padding='max_length', truncation=True, return_tensors="pt")
    input_ids, attention_mask = inputs['input_ids'].to(device), inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=-1)[0]
    
    prob_phishing = probs[1].item() * 100
    
    # 🔥 [점수 보정 1] 여백 채우기 (1-Depth 루트 URL)
    boost_weight_1 = 0.0
    if any(kw in processed_text for kw in high_risk_keywords): boost_weight_1 += 0.40
    if any(kw in processed_text for kw in action_keywords): boost_weight_1 += 0.15
    boost_weight_1 = min(boost_weight_1, 0.50) # 가중치 최대 50% 제한
    
    if boost_weight_1 > 0:
        prob_phishing += (100 - prob_phishing) * boost_weight_1 # 100% 안 넘게 남은 여백만큼만 더하기
    
    if prob_phishing > 50:
        return {"judgment": "unnormal", "riskLevel": "HIGH", "risklevel": "HIGH", "detectedUrl": target_url}

    # ----------------------------------------------------
    # 🌟 [2단계] 서브 링크 수집 (max_links = 2)
    # ----------------------------------------------------
    deep_links = extract_deep_links(raw_html, target_url, max_links=2)
    fetched_data = []

    if deep_links:
        def fetch_url_task(url):
            text, _ = extract_with_requests_and_raw_html(url)
            return url, text
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(fetch_url_task, url): url for url in deep_links}
            try:
                for future in concurrent.futures.as_completed(futures, timeout=3.0):
                    try: fetched_data.append(future.result())
                    except Exception: pass
            except concurrent.futures.TimeoutError:
                pass

    # ----------------------------------------------------
    # 🌟 [3단계] AI 일괄 병렬 검사
    # ----------------------------------------------------
    valid_urls = []
    valid_texts = []
    urls_to_pw_scan = [] 

    whitelist_domains = ["naver.com", "youtube.com", "daum.net", "kakao.com"]

    for idx, (url, current_text) in enumerate(fetched_data, 1):
        if any(safe_domain in url.lower() for safe_domain in whitelist_domains):
            continue 

        if len(current_text) < 150 or current_text.startswith("[오류]") or current_text.startswith("[판별 보류]"):
            if use_pw:
                urls_to_pw_scan.append(url) 
            continue

        has_risk = any(kw in current_text for kw in high_risk_keywords + action_keywords)
        if not current_text.startswith("[오류]"):
            if len(current_text) >= 80 or has_risk:
                valid_urls.append(url)
                valid_texts.append(current_text)

    # 비동기 병렬 처리 구역
    if urls_to_pw_scan:
        try:
            if not async_pw_manager:
                async_pw_manager = AsyncPlaywrightPool()
            
            pw_results = async_pw_manager.scrape_parallel(urls_to_pw_scan)
            
            for p_url, p_text, _ in pw_results:
                has_risk = any(kw in p_text for kw in high_risk_keywords + action_keywords)
                if not p_text.startswith("[오류]"):
                    if len(p_text) >= 80 or has_risk:
                        valid_urls.append(p_url)
                        valid_texts.append(p_text)
        except Exception:
            pass

    # AI 최종 추론 (Batch)
    if valid_texts:
        inputs = tokenizer(valid_texts, max_length=max_len, padding='max_length', truncation=True, return_tensors="pt")
        input_ids, attention_mask = inputs['input_ids'].to(device), inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1)
            
        for i, url in enumerate(valid_urls):
            current_text = valid_texts[i]
            prob_phishing = probs[i][1].item() * 100
            
            # 🔥 [점수 보정 2] 여백 채우기 (3-Depth 서브 링크)
            boost_weight_2 = 0.0
            if any(kw in current_text for kw in high_risk_keywords): boost_weight_2 += 0.40
            if any(kw in current_text for kw in action_keywords): boost_weight_2 += 0.15
            boost_weight_2 = min(boost_weight_2, 0.50)
            
            if boost_weight_2 > 0:
                prob_phishing += (100 - prob_phishing) * boost_weight_2
            
            if prob_phishing > 50:
                return {"judgment": "unnormal", "riskLevel": "HIGH", "risklevel": "HIGH", "detectedUrl": url}

    return {"judgment": "normal", "riskLevel": "LOW", "risklevel": "LOW", "detectedUrl": target_url}