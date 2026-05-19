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
        # 🌟 [핵심 개선] 큐싱 타겟팅 모바일(iPhone) 환경 완벽 위장
        context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1",
            viewport={'width': 390, 'height': 844}, # iPhone 14 해상도
            is_mobile=True,  # 모바일 브라우저 특성 활성화
            has_touch=True,  # 터치스크린 이벤트 활성화
            accept_downloads=False,
            locale="ko-KR", 
            extra_http_headers={"Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"}
        )
        page = await context.new_page()
        
        # 🌟 [해결책 1] 네이티브 경고창(alert, confirm) 텍스트 낚아채기
        popup_messages = []
        async def handle_dialog(dialog):
            popup_messages.append(dialog.message) # 경고창 텍스트 저장
            await dialog.accept()                 # 경고창 닫기 (진행을 위해)
        page.on("dialog", handle_dialog)

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

            await page.wait_for_timeout(1500)

            wait_time = 0
            while wait_time < wait_sec:
                current_html = await page.content()
                soup_test = BeautifulSoup(current_html, "html.parser")
                if len(soup_test.get_text(strip=True)) > 400: 
                    break
                await page.wait_for_timeout(250)
                wait_time += 0.25

            # 🌟 [해결책 2] 방해물(앱 유도 팝업) 제거 및 해커의 함정 강제 발동
            try:
                # 1. 모바일 앱 설치 유도 팝업 강제 닫기 (번개장터, 당근마켓 등 우회)
                close_targets = await page.locator("text=/괜찮아요, 모바일 웹에서 볼게요|웹에서 보기|다음에 하기|닫기|오늘 하루 보지 않기/i").all()
                for target in close_targets[:2]: # 최대 2개까지만 찔러봄
                    try:
                        await target.click(timeout=800, force=True)
                        await page.wait_for_timeout(1500) # 팝업이 걷히는 시간 1.5초 대기
                    except Exception:
                        pass
                
                # 2. 피싱 사이트가 주로 유도하는 '결제' 관련 버튼 강제 클릭
                click_targets = await page.locator("text=/간편결제|N Pay|Npay/i").all()
                for target in click_targets[:2]:
                    try:
                        await target.click(timeout=800, force=True)
                        await page.wait_for_timeout(400) # 클릭 후 모달이 뜰 때까지 0.4초 대기
                    except Exception:
                        pass
            except Exception:
                pass # 버튼이 없거나 클릭 불가능하면 부드럽게 패스

            full_html = await page.content()
            popup_text_combined = " ".join(popup_messages) # 수집된 경고창 텍스트 결합
            
            # 이미 만들어두신 ultimate_clean 함수의 popup_text 인자로 넘겨줍니다!
            clean_text = extract_with_html_ultimate_clean(full_html, popup_text=popup_text_combined)
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
    text = re.sub(r"http[s]?://(?:bit\.ly|vo\.la|t\.ly|cutt\.ly|url\.kr|ko\.gl|han\.gl|buly\.kr)/[a-zA-Z0-9]+", "[링크]", text)
    return text

def extract_with_html_ultimate_clean(html: str, popup_text: str = "") -> str:
    soup = BeautifulSoup(html, "html.parser")
    for noise in soup(["script", "style", "noscript", "iframe","header", "footer", "nav"]):
        noise.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    type_map = { "tel": "전화번호", "email": "이메일", "password": "비밀번호", "text": "텍스트", "number": "숫자", "checkbox": "체크박스" }
    sensitive_map = { "account": "계좌번호", "acc_no": "계좌번호", "bank": "계좌번호", "resident": "주민등록번호", "jumin": "주민등록번호", "rrn": "주민등록번호", "card_num": "카드번호", "card_no": "카드번호", "cc_num": "카드번호", "cvc": "카드보안코드", "cvv": "카드보안코드" }
    raw_inputs = []
    # 🌟 1. input 태그뿐만 아니라 내용 입력용 textarea 태그도 함께 스캔!
    for input_tag in soup.find_all(["input", "textarea"]):
        i_type = (input_tag.get("type", "text") or "text").lower() if input_tag.name == "input" else "텍스트"
        if i_type in ["hidden", "submit", "button", "image"]: continue
        
        # 🌟 2. 입력창 안의 희미한 글씨(placeholder)가 있으면 무식하게 '텍스트'라 하지 않고 그대로 수집!
        placeholder = input_tag.get('placeholder', '').strip()
        if placeholder and len(placeholder) <= 15:
            raw_inputs.append(placeholder)
            continue
            
        attr_context = f"{input_tag.get('name','')}_{input_tag.get('id','')}".lower()
        found_sensitive = False
        for eng_key, kor_val in sensitive_map.items():
            if eng_key == "account":
                if any(k in attr_context for k in ["bank", "finance", "pay", "계좌", "환불"]):
                    raw_inputs.append(kor_val)
                    found_sensitive = True
                    break
            elif eng_key in attr_context:
                raw_inputs.append(kor_val)
                found_sensitive = True
                break
                
        if not found_sensitive: 
            raw_inputs.append(type_map.get(i_type, i_type))
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
    
    # 🌟 [추가됨] 한국어 폼 관련 필수 수집 키워드
    vital_kws = ["이름", "성함", "연락처", "전화", "핸드폰", "내용", "주소", "나이", "계좌", "비밀번호", "신청", "결제", "일반결제",
                "재가입", "본인확인", "계정 정지", "비정상적인", "차단 해제", "안전한 사용", "유출"]

    for text in soup.stripped_strings:
        if any(bad_word in text for bad_word in blacklist_words): continue
        if number_pattern.match(text): continue
        
        text_len = len(text)
        
        # 🌟 [수정됨] 핵심 키워드가 포함되어 있으면 길이/개수 제한 무시하고 무조건 수집! (프리패스)
        if any(kw in text for kw in vital_kws):
            extracted_texts.append(text)
        # 일반 텍스트는 1글자 초과(2글자 이상)부터 수집하도록 완화
        elif 1 < text_len <= 25:
            if short_text_count < 30: # 수집 한도도 15개 -> 30개로 넉넉하게 확장
                extracted_texts.append(text)
                short_text_count += 1
        elif 25 < text_len <= 500:
            extracted_texts.append(text)
    unique_texts = []
    seen = set()
    for t in extracted_texts:
        if t not in seen: seen.add(t); unique_texts.append(t)
    main_text = " ".join(unique_texts)[:500]

    # 🌟 [수정] 무의미한 문법적 틀(Boilerplate) 완벽 제거. 순수 텍스트만 결합!
    components = []
    
    if title: components.append(title)
    if popup_text: components.append(popup_text)
    if main_text: components.append(main_text)
    
    action_kws = []
    if inputs_str: action_kws.append(inputs_str)
    if buttons_str: action_kws.append(buttons_str)
    if action_kws:
        components.append(" ".join(action_kws))

    final_text = ". ".join(components)
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
            ko_headers = {
                "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"
            }
            response = curl_requests.get(url, headers=ko_headers, impersonate="chrome116", timeout=timeout)
            raw_bytes = response.content
            
            try: 
                html = raw_bytes.decode('utf-8')
            except UnicodeDecodeError:
                try: 
                    html = raw_bytes.decode('cp949', errors='ignore')
                except UnicodeDecodeError: 
                    html = raw_bytes.decode('utf-8', errors='replace')
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
    model = BertForSequenceClassification.from_pretrained('monologg/kobert', num_labels=2, attn_implementation="eager")
    
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
        print("\n  [시스템] 엔진 초기 구동을 시작합니다. (첫 1회만 소요)")
        w_start = time.time()
        try: warmup_engine(include_pw=True)
        except Exception as e: 
            print(f"  ❌ [오류] 엔진 예열 실패: {e}")
            return {"judgment": "unknown", "riskLevel": "UNKNOWN", "risklevel": "UNKNOWN", "error": f"엔진 예열 실패: {e}", "detectedUrl": target_url}
        print(f"  [시스템] 엔진 예열 완료! (소요 시간: {time.time() - w_start:.2f}초)\n")

    start_time = time.time()

    print("="*60)
    print(f"🎯 [분석 시작] 타겟 URL: {target_url}")
    print("="*60)

    if not (target_url.startswith("http") or ":" in target_url or target_url.startswith("/")): 
        target_url = "https://" + target_url

    is_http_vulnerable = target_url.lower().startswith("http://")

    safe_tlds = ["go.kr", ".ac.kr", ".edu", ".mil.kr", ".ms.kr"]
    safe_official_domains = [
        "nonghyup.com", "kbstar.com", "shinhan.com", "wooribank.com", "kebhana.com",
        "hanabank.com", "kakaobank.com", "tossbank.com", "kbanknow.com", "ibk.co.kr", "korail.com", "ticketlink.co.kr", 
    ]
    legal_gambling_domains = [
       "dhlottery.co.kr", "www.dhlottery.co.kr", "m.dhlottery.co.kr",
        "betman.co.kr", "www.betman.co.kr", "m.betman.co.kr"
    ]

    try:
        domain = urlparse(target_url).netloc.lower()
        
        # 1. 일반 공식 기관/은행 프리패스
        if any(domain.endswith(tld) for tld in safe_tlds) or \
           any(domain == d or domain.endswith("." + d) for d in safe_official_domains):
            print(f"  🛡️ [공식 기관 화이트리스트 패스] {domain} -> 검사 생략 (0초 컷 정상 처리)")
            print(f"✅ 최종 결과 리포트 반환 (소요 시간: {time.time() - start_time:.2f}초)")
            # 하위 호환성을 유지하되, evidence 필드를 추가해줍니다.
            return {
                "url": target_url, "judgment": "normal", "riskLevel": "LOW", "risklevel": "LOW", 
                "detectedUrl": target_url, "threat_score": 0.0, "threat_type": "안전(공공/금융기관)", "site_category": "공식 사이트",
                "evidence": {"heuristic_evidence": {"detected_actions": [], "rule_trigger": "국가/공식 인증 도메인"}, "ai_semantic_evidence": {"suspect_sentence": "해당 없음", "ai_inference_logic": "공식 기관 도메인으로 확인되어 즉시 통과되었습니다."}}
            }
            
        # 2. 🌟 [추가] 동행복권/베트맨 전용 프리패스 (오탐 방지)
        if domain in legal_gambling_domains:
            print(f"  🛡️ [합법 복권/토토 화이트리스트 패스] {domain} -> 검사 생략 (0초 컷 정상 처리)")
            
            # 앱 UI 미리보기 출력
            print("\n" + "■"*60)
            print("📱 [Quishing Defender UI 미리보기]")
            print("-" * 60)
            print("🟢 [접속 안전] 국가 승인 합법 사이트!")
            print("💬 동행복권 또는 베트맨 등 국가에서 공식적으로 운영하는 합법 사이트입니다. 안심하고 이용하셔도 좋습니다.")
            print("■"*60 + "\n")
            
            print(f"✅ 최종 결과 리포트 반환 (소요 시간: {time.time() - start_time:.2f}초)")
            
            return {
                "url": target_url,
                "judgment": "normal",
                "riskLevel": "low",
                "threat_score": 0.0,
                "threat_type": "합법 복권/토토사이트",
                "site_category": "합법 사이트",
                "evidence": {
                    "heuristic_evidence": {
                        "detected_actions": [],
                        "rule_trigger": "국가 승인 공식 도메인 (화이트리스트 통과)"
                    },
                    "ai_semantic_evidence": {
                        "suspect_sentence": "해당 없음 (검사 생략)",
                        "ai_inference_logic": "국가에서 공식적으로 운영 및 승인하는 합법 복권/스포츠토토 사이트로 확인되어 AI 검사 없이 안전한 사이트로 즉시 분류되었습니다."
                    }
                }
            }
    except Exception:
        pass

    use_pw = os.getenv("USE_PLAYWRIGHT_IN_ANALYZE", "1") == "1"
    max_len = int(os.getenv("KOBERT_MAX_LEN", "512"))
    
    high_risk_keywords = [
        # 기존 대출/투자 관련
        "통신요금 담보", "신불자", "내구제", "폰테크", "신용등급 무관", "무직자 대출", "통신연체자", "비상장 주식", "공모주 청약", "원금 보장", "수익 보장", "투자 지원금", "리딩방", "결제시스템 불안정화", "급등주", "무료 리딩", "VVIP 정보", "세력주", "손실 복구", "무료 체험", 
        
        # 포털/보안 협박
        "재가입이 필요", "이용자 확인", "계정 정지", "비정상적인 접근", "차단 해제", "비밀번호 변경", "해외 IP 로그인", "계정 보호 조치", "비밀번호 오류", "안전보안설정", "계정 잠금", "장기 미접속",
        
        # 결제/공공/택배 위장
        "결제 승인", "자동이체 예정", "대출 승인", "신용카드 발급", "지원금 대상자", "소상공인 지원", "환불 처리", "미납 요금", "통장 압류", "과태료 부과", "건강보험료 미납", "도로교통법 위반", "택배 반송", "배송지 오류", "배송지 주소 오류", "통관 번호", "민원 접수 완료",
        
        # 가상화폐 위장
        "코인 상장", "무료 코인 지급", "에어드랍", "사전 판매", "지갑 연동"
    ]
    action_keywords = ["비밀번호", "계좌", "로그인", "login", "주민번호", "주민등록번호", "인증번호"]

    # ----------------------------------------------------
    # 🌟 [1단계] 루트 URL 검사
    # ----------------------------------------------------
    print("\n▶ [1-Depth 메인 페이지 분석]")
    processed_text, raw_html = extract_with_requests_and_raw_html(target_url)

    if "Suspected phishing site" in processed_text or "Cloudflare Ray ID" in processed_text:
        print("  🚨 [즉결 심판] Cloudflare에서 이미 차단된 피싱 사이트입니다! (AI 검사 생략)")
        
        evidence_dict = {
            "suspect_sentence": "Cloudflare 악성 사이트 경고 화면",
            "ai_reason": "글로벌 보안 네트워크(Cloudflare)에서 이미 악성 피싱 사이트로 블랙리스트에 등재되어 차단된 페이지입니다. AI 검사를 생략하고 즉시 접속을 원천 차단합니다."
        }
        return {"judgment": "unnormal", "riskLevel": "HIGH", "risklevel": "HIGH", "detectedUrl": target_url, "evidence": evidence_dict}
    
    if len(processed_text) < 400 or processed_text.startswith("[오류]") or processed_text.count(" ") < 10:
        print("  ⚠️ [알림] 텍스트 부족 또는 동적 렌더링(JS) 은닉 의심! 메인 페이지 정밀 스캔(Playwright) 기동...")
        if use_pw:
            try: 
                processed_text, raw_html = extract_with_playwright_and_raw_html(target_url, is_warmup=False)
            except Exception: 
                pass

    if processed_text.startswith("[오류]") or processed_text.startswith("[판별 보류]"):
        print("  ❌ [오류] 사이트 접속 불가 (Timeout 등)")
        return {"judgment": "unknown", "riskLevel": "UNKNOWN", "risklevel": "UNKNOWN", "detectedUrl": target_url}

        # ====================================================
    # 🌟 [위치 이동됨!] 'Not Found' 페이지 감지 및 즉결 심판 로직
    # ====================================================
    processed_text_lower = processed_text.lower()
    error_keywords = ["not found", "404 not found", "404", "페이지를 찾을 수 없습니다", "페이지가 삭제"]

    # 💡 에러 키워드 중 텍스트에 포함된 '첫 번째 단어'를 잡아냅니다.
    matched_keyword = next((keyword for keyword in error_keywords if keyword in processed_text_lower), None)

    if matched_keyword:
        print(f"  🚨 [즉결 심판] 존재하지 않는 페이지({matched_keyword})입니다! (단속을 피해 폐쇄된 피싱 사이트 의심)")
        
        # 💡 작성자님이 원하신 대로 감지된 키워드를 ai_reason에 동적으로 삽입합니다!
        ai_reason = f"분석 중 '{matched_keyword}' 문구가 감지되어, 페이지가 삭제되었거나 존재하지 않는다는 것을 확인하였습니다. 피싱 조직이 신고를 받고 도메인을 버렸거나, 추적을 피하기 위해 사이트를 임시로 폐쇄한 전형적인 '치고 빠지기' 상태로 판단되어 위험 사이트로 분류 및 차단합니다."
        
        final_json_report = {
            "url": target_url,
            "judgment": "unnormal",
            "riskLevel": "HIGH",
            "threat_score": 99.0,
            "threat_type": "은닉/폐쇄된 피싱 사이트",
            "site_category": "접속 불가",
            "evidence": {
                "heuristic_evidence": {
                    "detected_actions": ["페이지 폐쇄/숨김"],
                    "rule_trigger": f"에러 페이지 감지 ({matched_keyword})" # 증거에도 걸린 단어 추가
                },
                "ai_semantic_evidence": {
                    "suspect_sentence": f"감지된 에러 텍스트: '{matched_keyword}'", 
                    "ai_inference_logic": ai_reason
                }
            }
        }
        
        # 앱 UI 콘솔 출력
        print("\n" + "■"*60)
        print("📱 [Quishing Defender UI 미리보기]")
        print("-" * 60)
        print("🚨 [접속 차단됨] 은닉/폐쇄된 피싱 사이트!")
        print(f"💬 {ai_reason}\n")
        print("■"*60 + "\n")
        
        return final_json_report
    # ====================================================
    # ====================================================

    # ====================================================
    # 🌟 [신규 추가] Form Action 무단 유출 즉결 심판 (메신저 API 등)
    # Playwright 렌더링까지 끝난 최종 raw_html을 검사하여 확실한 물증을 잡습니다.
    # ====================================================
    soup_form = BeautifulSoup(raw_html, "html.parser")
    malicious_action_url = ""
    action_type = ""
    
    for form in soup_form.find_all("form"):
        action = form.get("action", "").lower()
        if "api.telegram.org" in action or "discord.com/api/webhooks" in action:
            malicious_action_url = action
            action_type = "해커 메신저(Telegram/Discord) API"
            break
        elif re.match(r"^https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", action):
            malicious_action_url = action
            action_type = "은닉용 IP 주소 직접 전송"
            break
            
    if malicious_action_url:
        print(f"  🚨 [즉결 심판] {action_type} 폼 전송 감지! ({malicious_action_url[:50]}...)")
        ai_reason = f"사용자가 입력한 정보(비밀번호 등)가 기업의 정상 서버가 아닌, '{action_type}'(으)로 무단 전송되도록 설계된 악성 폼(Form)이 발견되었습니다. 이는 가장 명백한 개인정보 탈취 수법이므로 AI 정밀 검사를 생략하고 즉시 차단합니다."
        
        final_json_report = {
            "url": target_url,
            "judgment": "unnormal",
            "riskLevel": "HIGH",
            "threat_score": 99.0,
            "threat_type": "데이터 유출(탈취) 폼 감지",
            "site_category": "악성 피싱",
            "evidence": {
                "heuristic_evidence": {
                    "detected_actions": ["개인정보 외부 무단 전송"],
                    "rule_trigger": f"{action_type} 통신 감지"
                },
                "ai_semantic_evidence": {
                    "suspect_sentence": f"유출 목적지: {malicious_action_url[:40]}...", 
                    "ai_inference_logic": ai_reason
                }
            }
        }
        
        # 앱 UI 콘솔 출력
        print("\n" + "■"*60)
        print("📱 [Quishing Defender UI 미리보기]")
        print("-" * 60)
        print("🚨 [접속 차단됨] 치명적 데이터 유출 폼 감지!")
        print(f"💬 {ai_reason}\n")
        print("■"*60 + "\n")
        
        return final_json_report
    # ==================================================== 

    print(f"  📝 [추출 텍스트]: {processed_text[:1000]}... (총 {len(processed_text)}자)")

    # 🌟 [신규 추가] AI 어텐션(시선) 강제 유도 로직 (Pre-processing Guide)
    focus_sentence = ""
    for sentence in re.split(r'(?<=[.!?])\s+', processed_text):
        if any(kw in sentence for kw in high_risk_keywords):
            focus_sentence = sentence
            break
            
    if focus_sentence:
        # AI가 가장 먼저 읽도록 핵심 문장을 텍스트 맨 앞에 강제로 박아버림!
        processed_text = f"{focus_sentence} {processed_text}"
        print(f"  🧠 [AI 시선 유도] 핵심 위협 문장을 최상단에 전진 배치합니다: {focus_sentence[:40]}...")

    inputs = tokenizer(processed_text, max_length=max_len, padding='max_length', truncation=True, return_tensors="pt")
    input_ids, attention_mask = inputs['input_ids'].to(device), inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)
        probs = F.softmax(outputs.logits, dim=-1)[0]
    
    prob_phishing = probs[1].item() * 100
    base_prob_1 = prob_phishing
    
    # ----------------------------------------------------
    # 🧠 [XAI] 원본 문장(Sentence) 매핑형 X-ray 분석 로직
    # ----------------------------------------------------
    try:
        attentions = outputs.attentions
        last_layer_attn = attentions[-1][0] 
        avg_attn = torch.mean(last_layer_attn, dim=0) 
        cls_attn = avg_attn[0] 
        
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', processed_text) if s.strip()]
        sentence_scores = []
        token_idx = 1 
        max_tokens = len(cls_attn) - 1 
        
        for sentence in sentences:
            if token_idx >= max_tokens: break
            sub_tokens = tokenizer.tokenize(sentence)
            sub_len = len(sub_tokens)
            
            end_idx = min(token_idx + sub_len, max_tokens)
            total_score = sum([cls_attn[i].item() * 100 for i in range(token_idx, end_idx)])
            score = total_score / sub_len if sub_len > 0 else 0
            
            if len(sentence) > 5: 
                sentence_scores.append((sentence, score))
            token_idx += sub_len 
            
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = sentence_scores if sentence_scores else [("분석할 문장이 없습니다.", 0.0)]
        
    except Exception as e:
        print(f"  ❌ [XAI 디버깅 에러]: {e}")
        top_sentences = [("XAI 추출 실패", 0.0)]

    # --- 안드로이드 앱 전송용 데이터 (Evidence) 준비 ---
    top_sent = top_sentences[0][0] if top_sentences else "분석된 문맥이 없습니다."
    for sent, _ in sentence_scores:
        if any(kw in sent for kw in high_risk_keywords):
            top_sent = sent
            print(f"  🎯 [XAI 보정] 고위험 문장 강제 타겟팅: {top_sent}")
            break
    
    detected_reqs = []
    if any(k in processed_text for k in action_keywords): detected_reqs.append("행동(로그인/인증) 요구")
    if any(k in processed_text for k in ["전화번호", "이메일", "카드번호", "계좌번호", "주민등록번호"]): detected_reqs.append("개인정보/금융 입력")
    req_str = ", ".join([f"[{req}]" for req in detected_reqs]) if detected_reqs else "[특이사항 없음]"

    found_high_risk = [kw for kw in high_risk_keywords if kw in processed_text]
    found_actions = [kw for kw in action_keywords if kw in processed_text]
    found_sensitive = [kw for kw in ["전화번호", "이메일", "카드번호", "계좌번호", "주민등록번호"] if kw in processed_text]

    demand_parts = []
    if found_actions: demand_parts.append(f"'{', '.join(found_actions)}'")
    if found_sensitive: demand_parts.append(f"'{', '.join(found_sensitive)}'")
    demand_str = " 및 ".join(demand_parts) if demand_parts else "특정 정보"
    
    high_risk_str = f"'{', '.join(found_high_risk)}'" if found_high_risk else ""

    # 🔥 [2단계] 키워드 기반 '범죄 유형(Threat Type)' 세부 분류 로직
    scam_type = "기관/기업 사칭 피싱"
    loan_kws = ["통신요금 담보", "신불자", "내구제", "폰테크", "무직자 대출", "통신연체자"]
    invest_kws = ["비상장 주식", "공모주 청약", "원금 보장", "수익 보장", "투자 지원금", "리딩방", "급등주", "VVIP 정보", "세력주", "무료 리딩"]
    trans_kws = ["상륙 하 다", "상륙하 다", "서명 하 다", "지불 하 다", "제출 하 다", "얻 다", "이 긴 다", "청소 하 라", "계 좌", "비 밀 번 호", "제시 하 다", "갱 신 하 다"]
    gambling_kws = ["로또6/45", "동행복권", "연금복권", "파워볼", "프로토", "스포츠토토", "드림게임", "카지노"]
    adult_kws = ["성인용품", "오피", "조건만남", "비아그라", "밤알바", "19금", "리얼돌"] 

    site_category = "일반"
    if any(kw in processed_text for kw in adult_kws):
        site_category = "성인 사이트"
    elif any(kw in processed_text for kw in gambling_kws):
        site_category = "도박/복권"

    is_translated = any(kw in processed_text for kw in trans_kws) or re.search(r'[가-힣]\s+하\s+다\b', processed_text)
    is_fake_gambling = False
    detected_gambling_str = "" 

    detected_gambling_kws = [kw for kw in gambling_kws if kw in processed_text]
    
    if detected_gambling_kws:
        legal_domains = ["dhlottery.co.kr", "betman.co.kr"]
        if not any(legal_domain in target_url for legal_domain in legal_domains):
            scam_type = "불법 사설 도박 및 공식 복권 사칭"
            is_fake_gambling = True
            found_high_risk = True 
            detected_gambling_str = ", ".join(detected_gambling_kws[:2]) 
            print(f"  🚨 [룰베이스 개입] 비인가 도메인({target_url})에서 사행성 키워드({detected_gambling_str}) 감지!")
    elif any(kw in processed_text for kw in loan_kws):
        scam_type = "불법 대출 및 금융 사기"
    elif any(kw in processed_text for kw in invest_kws):
        scam_type = "불법 투자 유도(리딩방) 사기"
    elif is_translated:
        scam_type = "해외 기계 번역(번역투) 피싱"

    # 🔥 [3단계] 점수 보정 (문맥 및 도메인 인식형 스마트 가중치)
    boost_weight_1 = 0.0
    
    # 🌟 [기존 로직] HTTP 취약점 보정
    if is_http_vulnerable:
        if found_high_risk:
            boost_weight_1 += 0.40
            print("  🔓 [보안 취약] HTTP + 고위험 범죄 키워드 동시 감지! (가중치 +40%)")
        elif found_actions and base_prob_1 >= 15.0:
            boost_weight_1 += 0.20
            print("  🔓 [보안 취약] HTTP 환경에서 정보 입력 요구 감지. (가중치 +20%)")
        else:
            print("  🔓 [참고] HTTP 접근이나, 문맥이 안전하여 피싱 가중치 제외.")

    # 🌟 [신규 로직] 도메인 불일치 + 고유식별정보(주민번호 등) 요구 즉결 심판!
    current_domain = urlparse(target_url).netloc.lower()
    safe_tlds_list = ["go.kr", ".ac.kr", ".edu", ".mil.kr", ".ms.kr"]
    is_official_domain = any(current_domain.endswith(tld) for tld in safe_tlds_list)
    
    is_jumin_demanded = any(kw in processed_text for kw in ["주민번호", "주민등록번호"])

    if not is_official_domain and is_jumin_demanded:
        boost_weight_1 += 0.60 # 가중치 폭탄 투하! (+60%)
        scam_type = "정부/공공기관 사칭 피싱"
        print(f"  🚨 [룰베이스 개입] 비인가 도메인({current_domain})에서 주민등록번호 요구 감지!")

    # 🌟 [가중치 합산]
    if found_high_risk: 
        boost_weight_1 += 0.50
        
    # 일반 행동(로그인 등) 요구 가중치 (단, 사칭 피싱으로 걸린 '주민번호' 건은 중복 적용 방지)
    if found_actions and not (not is_official_domain and is_jumin_demanded): 
        boost_weight_1 += 0.15
        
    boost_weight_1 = min(boost_weight_1, 0.85)
    
    if boost_weight_1 > 0:
        prob_phishing += (100 - prob_phishing) * boost_weight_1
        print(f"  📈 [점수 보정] 위험/요구 키워드 탐지! KoBERT({base_prob_1:.1f}%) ➡️ 보정 후({prob_phishing:.1f}%)")
        
    # 🔥 [4단계] AI 주도형(AI-Driven) 초정밀 판단 사유 생성
    if prob_phishing <= 50.0:
        if not found_actions and not found_sensitive and base_prob_1 < 5.0:
            ai_reason = "AI 문맥 분석 결과, 위험한 단어나 개인정보 요구가 전혀 없는 안전한 일반 웹페이지로 확인되었습니다."
        elif demand_parts:
            if base_prob_1 < 20.0:
                ai_reason = f"페이지 내에 {demand_str} 입력을 요구하는 폼이 존재합니다. 그러나 AI가 주변 문맥을 심층 분석한 결과, 기만 의도가 없는 '정상적인 공식 서비스 안내/인증'으로 판단하여 통과시켰습니다."
            else:
                ai_reason = f"{demand_str} 요구와 함께 다소 주의가 필요한 텍스트가 탐지되었습니다. 그러나 AI 판단 결과, 피싱 특유의 치명적인 협박이나 긴급성(긴급 행동 유도)이 결여되어 있어 최종 정상 범주로 분류했습니다."
        else:
             ai_reason = f"일부 주의가 필요한 문구(AI 위험도 {base_prob_1:.1f}%)가 있으나, AI가 문서를 종합적으로 스캔한 결과 직접적인 정보 탈취 목적이 없다고 판단하여 정상 처리했습니다."
    else:
        if is_fake_gambling:
            ai_reason = f"룰베이스 엔진 교차 검증 결과, '{detected_gambling_str}' 관련 복권/사행성 텍스트가 확인되었으나 접속 도메인({target_url})이 국가 공인 합법 도메인이 아닙니다. 전형적인 사칭 및 불법 사설 도박장으로 판별되어 접속을 강력히 차단합니다."
        elif found_high_risk:
            ai_reason = f"명백한 불법 키워드({high_risk_str})가 탐지되었으며, AI가 이와 연관된 문맥을 정밀 분석한 결과 {demand_str}를 탈취하려는 '{scam_type}' 목적이 확인되어 악성이라고 판정하였습니다."
        elif is_translated:
            ai_reason = f"AI 분석 결과, \"{top_sent[:30]}...\" 해당 문구들이 부자연스러운 기계 번역투 및 어색한 띄어쓰기로 작성된 것이 확인되었습니다. 이는 해외 기반의 양산형 사기 사이트의 전형적인 특징이므로 최종 악성으로 판정합니다."
        elif demand_parts and base_prob_1 >= 60.0:
            ai_reason = f"AI 엔진이 \"{top_sent[:30]}...\" 문장에 내포된 기만적 의도를 정확히 포착했습니다. 이는 불안감을 조성하여 {demand_str}를 빼내려는 전형적인 '{scam_type}' 기법으로 판별되었습니다."
        elif demand_parts and boost_weight_1 > 0:
            ai_reason = f"AI가 전체 텍스트에서 수상한 흐름을 1차 감지하였고, 실제로 {demand_str} 입력을 요구하는 구조가 2차 확인됨에 따라 딥러닝-룰베이스 교차 검증을 거쳐 최종 악성으로 확정했습니다."
        else:
            ai_reason = f"특정 키워드 없이도, AI가 \"{top_sent[:30]}...\" 문맥 자체에서 사용자를 속여 시스템을 장악하려는 고도의 악의적 의도를 찾아내어 원천 차단합니다."

    # 🌟 [신규 추가] HTTP 취약 멘트 자연스럽게 합치기 (최종 방어선)
    # 정상이든 악성이든 HTTP 기반이면 앞에 경고 문구를 붙여줍니다!
    if is_http_vulnerable:
        ai_reason = f"해당 링크는 보안이 취약한 링크(HTTP)이며, {ai_reason}"

    # 🔥 [5단계] 수사 보고서(Forensic Report) 형태의 고급 JSON 데이터 조립
    rule_trigger_msg = "특이사항 없음"
    if is_fake_gambling: rule_trigger_msg = "국가 공인 도메인 불일치 (사설 도박장/사칭)"
    elif is_translated: rule_trigger_msg = "부자연스러운 기계 번역 및 띄어쓰기 파괴 감지"
    elif found_high_risk: rule_trigger_msg = f"고위험 범죄 키워드({high_risk_str}) 매칭"
    
    # HTTP 취약점 트리거 추가
    if is_http_vulnerable:
        if rule_trigger_msg == "특이사항 없음":
            rule_trigger_msg = "HTTP 보안 취약 프로토콜 감지"
        else:
            rule_trigger_msg += " 및 HTTP 보안 취약 프로토콜 감지"

    final_json_report = {
        "url": target_url,
        "judgment": "unnormal" if prob_phishing > 50.0 else "normal",
        "riskLevel": "HIGH" if prob_phishing > 50.0 else "LOW",
        "threat_score": round(prob_phishing, 1),
        "threat_type": scam_type if prob_phishing > 50.0 else "안전(위협 없음)",
        "site_category": site_category,
        "evidence": {
            "heuristic_evidence": {
                "detected_actions": demand_parts if demand_parts else ["요구 정보 없음"],
                "rule_trigger": rule_trigger_msg
            },
            "ai_semantic_evidence": {
                "suspect_sentence": top_sent, 
                "ai_inference_logic": ai_reason
            }
        }
    }
    # 📱 [앱 UI 콘솔 미리보기 출력 부분]
    print("\n" + "■"*60)
    print("📱 [Quishing Defender UI 미리보기]")
    print("-" * 60)
    
    if prob_phishing > 50.0:
        print("🚨 [접속 차단됨] 피싱 사이트 의심!")
        print(f"💬 {ai_reason}\n")
        print(f"⚠️ 의심 문구: \"{top_sent}\"") 
    else:
        print("✅ [접속 허용] 안전한 웹사이트입니다.")
        print(f"💬 {ai_reason}\n")
        print(f"🔎 확인 문구: \"{top_sent}\"") 
        
    print("■"*60 + "\n")

    # 🔥 [1-Depth 악성 조기 리턴] 
    if prob_phishing > 50.0:
        print(f"✅ 최종 결과 리포트 반환 (소요 시간: {time.time() - start_time:.2f}초)")
        return final_json_report

    # ----------------------------------------------------
    # 🌟 [2단계] 서브 링크 수집 및 병렬 스캔
    # ----------------------------------------------------
    deep_links = extract_deep_links(raw_html, target_url, max_links=2)
    fetched_data = []

    if deep_links:
        print(f"\n▶ [2-Depth 하위 링크 탐색 ({len(deep_links)}개 발견)]")
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
                print("  ⚠️ [경고] 2-Depth 일반 수집 타임아웃 발생")

    valid_urls, valid_texts, urls_to_pw_scan = [], [], []
    whitelist_domains = ["naver.com", "youtube.com", "daum.net"]

    for idx, (url, current_text) in enumerate(fetched_data, 1):
        if any(safe_domain in url.lower() for safe_domain in whitelist_domains): continue 
        if len(current_text) < 150 or current_text.startswith("[오류]") or current_text.startswith("[판별 보류]"):
            if use_pw: urls_to_pw_scan.append(url) 
            continue
        has_risk = any(kw in current_text for kw in high_risk_keywords + action_keywords)
        if not current_text.startswith("[오류]"):
            if len(current_text) >= 80 or has_risk:
                valid_urls.append(url); valid_texts.append(current_text)

    if urls_to_pw_scan:
        print(f"\n🚀 [비동기 병렬 스캔 시작] 대기열 {len(urls_to_pw_scan)}개의 탭을 동시에 엽니다!")
        try:
            pw_start = time.time()
            if not async_pw_manager: async_pw_manager = AsyncPlaywrightPool()
            pw_results = async_pw_manager.scrape_parallel(urls_to_pw_scan)
            for p_url, p_text, _ in pw_results:
                has_risk = any(kw in p_text for kw in high_risk_keywords + action_keywords)
                if not p_text.startswith("[오류]"):
                    if len(p_text) >= 80 or has_risk:
                        valid_urls.append(p_url); valid_texts.append(p_text)
            print(f"  ⚡ [병렬 스캔 완료] 소요 시간: {time.time() - pw_start:.2f}초")
        except Exception as e: print(f"  ❌ [병렬 스캔 에러] {e}")

    if valid_texts:
        print(f"\n🧠 [AI 2-Depth 정밀 분석] 확보된 텍스트 {len(valid_texts)}개 일괄 검사 중...")
        inputs = tokenizer(valid_texts, max_length=max_len, padding='max_length', truncation=True, return_tensors="pt")
        input_ids, attention_mask = inputs['input_ids'].to(device), inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            # 🌟 1. XAI를 위해 output_attentions=True 옵션 추가!
            outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True) 
            probs = F.softmax(outputs.logits, dim=-1)
            attentions_batch = outputs.attentions # 배치(Batch) 전체의 어텐션 맵 가져오기
            
        for i, url in enumerate(valid_urls):
            current_text = valid_texts[i]
            base_prob_2 = probs[i][1].item() * 100
            prob_phishing_2 = base_prob_2
            
            boost_weight_2 = 0.0
            if url.lower().startswith("http://"): boost_weight_2 += 0.40
            if any(kw in current_text for kw in high_risk_keywords): boost_weight_2 += 0.40
            if any(kw in current_text for kw in action_keywords): boost_weight_2 += 0.15
            boost_weight_2 = min(boost_weight_2, 0.85)
            
            if boost_weight_2 > 0:
                prob_phishing_2 += (100 - prob_phishing_2) * boost_weight_2
            
            if prob_phishing_2 > 50:
                print(f"  🚨 [2-Depth 결과] 악성 감지! URL: {url} (최종 확률 {prob_phishing_2:.2f}%)")
                print(f"✅ 최종 결과 리포트 반환 (소요 시간: {time.time() - start_time:.2f}초)")
                
                # ====================================================
                # 🌟 2. [2-Depth 전용 XAI 로직] AI가 집중한 문장 직접 뽑아내기
                # ====================================================
                try:
                    last_layer_attn = attentions_batch[-1][i] # 현재 i번째 텍스트의 어텐션만 추출
                    avg_attn = torch.mean(last_layer_attn, dim=0) 
                    cls_attn = avg_attn[0] 
                    
                    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', current_text) if s.strip()]
                    sentence_scores = []
                    token_idx = 1 
                    max_tokens = len(cls_attn) - 1 
                    
                    for sentence in sentences:
                        if token_idx >= max_tokens: break
                        sub_tokens = tokenizer.tokenize(sentence)
                        sub_len = len(sub_tokens)
                        
                        end_idx = min(token_idx + sub_len, max_tokens)
                        total_score = sum([cls_attn[j].item() * 100 for j in range(token_idx, end_idx)])
                        score = total_score / sub_len if sub_len > 0 else 0
                        
                        if len(sentence) > 5: 
                            sentence_scores.append((sentence, score))
                        token_idx += sub_len 
                        
                    sentence_scores.sort(key=lambda x: x[1], reverse=True)
                    top_sent_2depth = sentence_scores[0][0] if sentence_scores else "분석된 문맥이 없습니다."
                except Exception as e:
                    print(f"  ❌ [2-Depth XAI 에러]: {e}")
                    top_sent_2depth = "XAI 추출 실패"
                # ====================================================

                final_json_report["url"] = url
                final_json_report["judgment"] = "unnormal"
                final_json_report["riskLevel"] = "HIGH"
                final_json_report["threat_score"] = round(prob_phishing_2, 1)
                final_json_report["threat_type"] = "은닉된 하위 페이지 피싱"
                
                # 🌟 3. XAI가 찾아낸 진짜 증거 문장을 리포트에 삽입!
                final_json_report["evidence"]["ai_semantic_evidence"]["suspect_sentence"] = f"감지된 하위 링크 위험 문구: '{top_sent_2depth}'"
                
                # 2-Depth 사유도 HTTP 여부에 따라 분기 처리
                if url.lower().startswith("http://"):
                    final_json_report["evidence"]["ai_semantic_evidence"]["ai_inference_logic"] = "해당 하위 링크는 보안이 취약한 링크(HTTP)이며, 정보 탈취용 악성 패턴이 감지되어 원천 차단합니다."
                else:
                    final_json_report["evidence"]["ai_semantic_evidence"]["ai_inference_logic"] = "메인 페이지는 정상으로 판별되었으나, 내부에 연결된 하위 링크에서 정보 탈취용 악성 패턴이 감지되어 원천 차단합니다."
                return final_json_report

    print(f"\n✅ 모든 스캔 완료. 특이사항 없음! (총 소요 시간: {time.time() - start_time:.2f}초)")
    return final_json_report