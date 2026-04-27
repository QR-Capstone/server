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

    # 🔥 원래의 고속 스캔 속도(6000ms)로 원상 복구!
    async def _fetch_single(self, url, timeout_ms=6000, wait_sec=3.5):
        # 안티봇 회피를 위해 평범한 윈도우 크롬 사용자인 척 위장은 유지
        context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={'width': 1920, 'height': 1080}
        )
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

def extract_with_html_ultimate_clean(html: str, popup_text: str = "") -> str:
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
            # 🔥 [수정] account 단어의 오탐(User Account) 방지 로직 추가
            if eng_key == "account":
                # bank, finance 등의 단어가 함께 있거나, 한국어 placeholder에 '계좌'가 있을 때만 '계좌번호'로 인정
                if any(k in attr_context for k in ["bank", "finance", "pay", "계좌", "환불"]):
                    raw_inputs.append(kor_val)
                    found_sensitive = True
                    break
            elif eng_key in attr_context:
                raw_inputs.append(kor_val)
                found_sensitive = True
                break
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
    if popup_text:context_sentences.append(f"접속 시 화면에 다음과 같은 경고 팝업이 발생했습니다. {popup_text}")
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
            # 🔥 속도 저하를 막기 위해 원래의 칼차단 1.2초로 복구!
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

    safe_tlds = [".go.kr", ".ac.kr", ".edu", ".mil.kr", ".ms.kr"]
    safe_official_domains = [
        "nonghyup.com", "kbstar.com", "shinhan.com", "wooribank.com",
        "hanabank.com", "kakaobank.com", "tossbank.com", "kbanknow.com", "ibk.co.kr", "korail.com", "ticketlink.co.kr"
    ]

    try:
        domain = urlparse(target_url).netloc.lower()
        if any(domain.endswith(tld) for tld in safe_tlds) or \
           any(domain == d or domain.endswith("." + d) for d in safe_official_domains):
            print(f"  🛡️ [공식 기관 화이트리스트 패스] {domain} -> 검사 생략 (0초 컷 정상 처리)")
            print(f"✅ 최종 결과 리포트 반환 (소요 시간: {time.time() - start_time:.2f}초)")
            return {"judgment": "normal", "riskLevel": "LOW", "risklevel": "LOW", "detectedUrl": target_url}
    except Exception:
        pass

    use_pw = os.getenv("USE_PLAYWRIGHT_IN_ANALYZE", "1") == "1"
    max_len = int(os.getenv("KOBERT_MAX_LEN", "512"))
    
    high_risk_keywords = ["통신요금 담보", "신불자", "내구제", "폰테크", "신용등급 무관", "무직자 대출", "통신연체자", "비상장 주식", "공모주 청약", "원금 보장", "수익 보장", "투자 지원금", "리딩방", "네이버pay 사용이 불가능", "결제시스템 불안정화", "급등주", "무료 리딩", "VVIP 정보", "세력주", "손실 복구", "무료 체험"]
    action_keywords = ["비밀번호", "계좌", "로그인", "login", "주민번호", "주민등록번호", "인증번호", "입력을 요구"]

    # ----------------------------------------------------
    # 🌟 [1단계] 루트 URL 검사
    # ----------------------------------------------------
    print("\n▶ [1-Depth 메인 페이지 분석]")
    processed_text, raw_html = extract_with_requests_and_raw_html(target_url)

    if "Suspected phishing site" in processed_text or "Cloudflare Ray ID" in processed_text:
        print("  🚨 [즉결 심판] Cloudflare에서 이미 차단된 피싱 사이트입니다! (AI 검사 생략)")
        
        # 🔥 [추가] 즉결 심판 시에도 안드로이드 앱에서 텍스트를 띄울 수 있도록 evidence 추가
        evidence_dict = {
            "suspect_sentence": "Cloudflare 악성 사이트 경고 화면",
            "ai_reason": "글로벌 보안 네트워크(Cloudflare)에서 이미 악성 피싱 사이트로 블랙리스트에 등재되어 차단된 페이지입니다. AI 검사를 생략하고 즉시 접속을 원천 차단합니다."
        }
        return {"judgment": "unnormal", "riskLevel": "HIGH", "risklevel": "HIGH", "detectedUrl": target_url, "evidence": evidence_dict}
    
    if len(processed_text) < 150 or processed_text.startswith("[오류]"):
        print("  ⚠️ [알림] 텍스트 부족/오류 감지! 메인 페이지 정밀 스캔(Playwright) 기동...")
        if use_pw:
            try: processed_text, raw_html = extract_with_playwright_and_raw_html(target_url, is_warmup=False)
            except Exception: pass

    if processed_text.startswith("[오류]") or processed_text.startswith("[판별 보류]"):
        print("  ❌ [오류] 사이트 접속 불가 (Timeout 등)")
        return {"judgment": "unknown", "riskLevel": "UNKNOWN", "risklevel": "UNKNOWN", "detectedUrl": target_url}

    print(f"  📝 [추출 텍스트]: {processed_text[:1000]}... (총 {len(processed_text)}자)")

    inputs = tokenizer(processed_text, max_length=max_len, padding='max_length', truncation=True, return_tensors="pt")
    input_ids, attention_mask = inputs['input_ids'].to(device), inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        # 🔥 [핵심 추가] output_attentions=True 를 넣어야 XAI 추출이 가능합니다!
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
            score = sum([cls_attn[i].item() * 100 for i in range(token_idx, end_idx)])
            
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
    top_score = top_sentences[0][1] if top_sentences else 0.0
    
    detected_reqs = []
    if any(k in processed_text for k in action_keywords): detected_reqs.append("행동(로그인/인증) 요구")
    if any(k in processed_text for k in ["전화번호", "이메일", "카드번호", "계좌번호", "주민등록번호"]): detected_reqs.append("개인정보/금융 입력")
    req_str = ", ".join([f"[{req}]" for req in detected_reqs]) if detected_reqs else "[특이사항 없음]"

    # 🔥 [1단계] 텍스트에서 실제로 발견된 위험/요구 '키워드' 추출
    found_high_risk = [kw for kw in high_risk_keywords if kw in processed_text]
    found_actions = [kw for kw in action_keywords if kw in processed_text]
    found_sensitive = [kw for kw in ["전화번호", "이메일", "카드번호", "계좌번호", "주민등록번호"] if kw in processed_text]

    # 발견된 단어들을 예쁜 자연어로 묶기
    demand_parts = []
    if found_actions: demand_parts.append(f"'{', '.join(found_actions)}'")
    if found_sensitive: demand_parts.append(f"'{', '.join(found_sensitive)}'")
    demand_str = " 및 ".join(demand_parts) if demand_parts else "특정 정보"
    
    high_risk_str = f"'{', '.join(found_high_risk)}'" if found_high_risk else ""

    # 🔥 [2단계] 키워드 기반 '범죄 유형(Threat Type)' 세부 분류 로직
    scam_type = "기관/기업 사칭 피싱" # 기본값
    loan_kws = ["통신요금 담보", "신불자", "내구제", "폰테크", "무직자 대출", "통신연체자"]
    invest_kws = ["비상장 주식", "공모주 청약", "원금 보장", "수익 보장", "투자 지원금", "리딩방", "급등주", "VVIP 정보", "세력주", "무료 리딩"]
    
    # 🌟 [추가] 어색한 기계 번역투 감지 키워드 및 정규식 로직
    trans_kws = ["상륙 하 다", "상륙하 다", "서명 하 다", "지불 하 다", "제출 하 다", "얻 다", "이 긴 다", "청소 하 라", "계 좌", "비 밀 번 호", "제시 하 다", "갱 신 하 다"]
    # 정규식: "동사 + 하 다" 형태로 비정상적으로 띄어쓰기가 파괴된 경우 감지
    is_translated = any(kw in processed_text for kw in trans_kws) or re.search(r'[가-힣]\s+하\s+다\b', processed_text)

    if any(kw in processed_text for kw in loan_kws):
        scam_type = "불법 대출 및 금융 사기"
    elif any(kw in processed_text for kw in invest_kws):
        scam_type = "불법 투자 유도(리딩방) 사기"
    elif is_translated:
        scam_type = "해외 기계 번역(번역투) 피싱" # 번역투 사기 유형으로 확정!

    # 🔥 [3단계] 점수 보정 
    boost_weight_1 = 0.0
    if found_high_risk: boost_weight_1 += 0.50
    if found_actions: boost_weight_1 += 0.15
    boost_weight_1 = min(boost_weight_1, 0.75) 
    
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
        if found_high_risk:
            ai_reason = f"명백한 불법 키워드({high_risk_str})가 탐지되었으며, AI가 이와 연관된 문맥을 정밀 분석한 결과 {demand_str}를 탈취하려는 '{scam_type}' 목적이 확실시되어 접속을 차단합니다."
        elif is_translated:
            # 🌟 [추가] 번역투 감지 시 전용 메세지 출력!
            ai_reason = f"AI 분석 결과, \"{top_sent[:20]}...\" 해당 문구들이 부자연스러운 기계 번역투 및 어색한 띄어쓰기로 작성된 것이 확인되었습니다. 이는 해외 기반의 양산형 사기 사이트의 전형적인 특징이므로 최종 악성으로 판별 및 차단합니다."
        elif demand_parts and base_prob_1 >= 60.0:
            ai_reason = f"AI 엔진이 \"{top_sent[:25]}...\" 문장에 내포된 기만적 의도를 정확히 포착했습니다. 이는 불안감을 조성하여 {demand_str}를 빼내려는 전형적인 '{scam_type}' 기법으로 판별되었습니다."
        elif demand_parts and boost_weight_1 > 0:
            ai_reason = f"AI가 전체 텍스트에서 수상한 흐름(위험도 {base_prob_1:.1f}%)을 1차 감지하였고, 실제로 {demand_str} 입력을 요구하는 구조가 2차 확인됨에 따라 딥러닝-룰베이스 교차 검증을 거쳐 최종 악성으로 확정했습니다."
        else:
            ai_reason = f"특정 키워드 없이도, AI가 \"{top_sent[:20]}...\" 문맥 자체에서 사용자를 속여 시스템을 장악하려는 고도의 악의적 의도를 찾아내어 원천 차단합니다."

    # 🔥 최종 evidence_dict 생성
    evidence_dict = {
        "suspect_sentence": top_sent, 
        "ai_reason": ai_reason
    }
   # 📱 [앱 UI 콘솔 미리보기 출력 부분도 정제]
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

    # 🔥 [여기에 버그 픽스 코드 추가!] 1-Depth에서 악성(50% 초과)으로 확정 났다면, 2-Depth로 가지 않고 즉시 악성 리턴!
    if prob_phishing > 50.0:
        print(f"✅ 최종 결과 리포트 반환 (소요 시간: {time.time() - start_time:.2f}초)")
        return {
            "judgment": "unnormal", 
            "riskLevel": "HIGH", 
            "risklevel": "HIGH", 
            "detectedUrl": target_url, 
            "evidence": evidence_dict
        }

    # ----------------------------------------------------
    # 🌟 [2단계] 서브 링크 수집 및 병렬 스캔 (기존 로직 유지)
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
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1)
            
        for i, url in enumerate(valid_urls):
            current_text = valid_texts[i]
            base_prob_2 = probs[i][1].item() * 100
            prob_phishing_2 = base_prob_2
            
            boost_weight_2 = 0.0
            if any(kw in current_text for kw in high_risk_keywords): boost_weight_2 += 0.40
            if any(kw in current_text for kw in action_keywords): boost_weight_2 += 0.15
            boost_weight_2 = min(boost_weight_2, 0.50)
            
            if boost_weight_2 > 0:
                prob_phishing_2 += (100 - prob_phishing_2) * boost_weight_2
            
            if prob_phishing_2 > 50:
                print(f"  🚨 [2-Depth 결과] 악성 감지! URL: {url} (최종 확률 {prob_phishing_2:.2f}%)")
                print(f"✅ 최종 결과 리포트 반환 (소요 시간: {time.time() - start_time:.2f}초)")
                # 🔥 evidence 리턴 추가
                return {"judgment": "unnormal", "riskLevel": "HIGH", "risklevel": "HIGH", "detectedUrl": url, "evidence": evidence_dict}

    print(f"\n✅ 모든 스캔 완료. 특이사항 없음! (총 소요 시간: {time.time() - start_time:.2f}초)")
    # 🔥 evidence 리턴 추가
    return {"judgment": "normal", "riskLevel": "LOW", "risklevel": "LOW", "detectedUrl": target_url, "evidence": evidence_dict}