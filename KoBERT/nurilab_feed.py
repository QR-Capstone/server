"""Compare only recent Nurilab alerts, using memory without a disk database.

KOBERT_NURILAB_DAYS=5..7 (default 7), including today in Asia/Seoul.
Never connects to any reported threat URL. CLI --refresh is a connectivity check.
"""
import argparse
from datetime import datetime, timezone, timedelta, date
from html.parser import HTMLParser
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit, urlunsplit, urlencode
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError

SOURCE = 'https://www.nurilab.com/kr/alert/phishing_alert'
ENDPOINT = 'https://www.nurilab.com/kr/alert/paginate_alerts'
KST = timezone(timedelta(hours=9))
REFRESH_SECONDS = 3600
FETCH_BUDGET = 12.0
ASKURL_BUDGET = 20.0
ASKURL_TIMEOUT = 4.0
INITIAL_REFRESH_TIMEOUT = FETCH_BUDGET + ASKURL_BUDGET + ASKURL_TIMEOUT + 1.0
_ASK_RESULT_RE = re.compile(r'https://askurl\.io/urls/([a-fA-F0-9]{64})/?\Z')
_DETAIL_RE = re.compile(r'https://www\.nurilab\.com/kr/alert/phishing_alert/phishing_alert_paper/(20\d{2})/phishing_alert_paper(20\d{6})_\d+_\d+\.html\Z')
_ask_cache_lock = threading.Lock()
_ask_cache = {}
_ask_retry_until = 0.0
_ask_backoff_seconds = 300.0
_last_refresh_status = {}
# Prioritize registered short links, whose destination is absent from the direct index.
_SHORT_LINK_HOSTS = {'sw.run', 'bit.ly', 't.co', 'tinyurl.com', 'vo.la', 't.ly',
                     'cutt.ly', 'url.kr', 'ko.gl', 'han.gl', 'buly.kr', 'me2.kr'}
DETECTION_KINDS = ('피싱', '사기')
_lock = threading.Lock()
_refresh_lock = threading.Lock()
_index = {}
_expires_at = 0.0
_loaded_days = None
_next_retry = 0.0
_worker_lock = threading.Lock()
_worker = None
_stop_event = None
_first_attempt = None


def canonical_url(value):
    """Conservative exact URL key: preserve scheme, path, query and fragment."""
    if not isinstance(value, str):
        return None
    value = value.strip().replace('[.]', '.').replace('[://]', '://')
    value = re.sub(r'^hxxps:', 'https:', value, flags=re.I)
    value = re.sub(r'^hxxp:', 'http:', value, flags=re.I)
    if not value or '...' in value or '…' in value or re.search(r'[\s\x00-\x1f\\]', value):
        return None
    if '://' not in value:
        value = 'https://' + value
    try:
        parts = urlsplit(value)
        if parts.scheme.lower() not in ('http', 'https') or not parts.hostname or parts.username or parts.password:
            return None
        host = parts.hostname.rstrip('.').encode('idna').decode('ascii').lower()
        if ':' in host:
            host = '[' + host + ']'
        port = parts.port
        scheme = parts.scheme.lower()
        authority = host if port is None or (scheme, port) in (('http', 80), ('https', 443)) else f'{host}:{port}'
        return urlunsplit((scheme, authority, parts.path or '/', parts.query, parts.fragment))
    except (ValueError, UnicodeError):
        return None


class AlertParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.row = None
        self.rows = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get('class', '').split()
        if tag in ('br', 'img', 'meta', 'link', 'input', 'hr', 'wbr', 'source'):
            return
        self.stack.append((tag, classes))
        if 'page_content' in classes:
            self.row = {'kind': attrs.get('data-detect', ''), 'title': '', 'url': '', 'date': '', 'depth': len(self.stack)}
        if self.row and 'content_target_item' in classes:
            self.row['url'] = attrs.get('title', '').strip()
        if self.row and tag == 'a' and 'load-content-phishing' in classes:
            year, filename = attrs.get('data-path', ''), attrs.get('data-file', '')
            detail = f'https://www.nurilab.com/kr/alert/phishing_alert/phishing_alert_paper/{year}/{filename}'
            if _DETAIL_RE.fullmatch(detail):
                self.row['detail_url'] = detail

    def handle_data(self, data):
        if not self.row:
            return
        active = {cls for _, classes in self.stack[self.row['depth']:] for cls in classes}
        if 'content_title_item' in active:
            self.row['title'] += data
        if 'content_date_item' in active:
            self.row['date'] += data

    def handle_endtag(self, tag):
        if not self.stack:
            return
        position = next((i for i in range(len(self.stack)-1, -1, -1) if self.stack[i][0] == tag), None)
        if position is None:
            return
        if self.row and position < self.row['depth']:
            self.rows.append(self.row)
            self.row = None
        del self.stack[position:]


def parse_alerts(html):
    parser = AlertParser()
    parser.feed(html)
    if not parser.rows:
        raise ValueError('누리랩 목록 구조를 확인할 수 없습니다')
    entries = []
    for row in parser.rows:
        kind = row['kind'].strip()
        if kind not in DETECTION_KINDS:
            continue
        key = canonical_url(row['url'])
        if key:
            entry = {'url': key, 'title': ' '.join(row['title'].split()),
                     'published_date': ' '.join(row['date'].split()), 'kind': kind}
            if row.get('detail_url'):
                entry['detail_url'] = row['detail_url']
            entries.append(entry)
    return entries


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('목록 요청의 리다이렉트는 허용하지 않습니다')


class _AskLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        href = dict(attrs).get('href', '')
        if tag == 'a' and _ASK_RESULT_RE.fullmatch(href) and href not in self.links:
            if len(self.links) < 8:
                self.links.append(href)


class AskRateLimited(OSError):
    pass


def _ask_retry_seconds():
    with _ask_cache_lock:
        return max(0.0, _ask_retry_until - time.monotonic())


def _note_rate_limit(retry_after):
    global _ask_retry_until, _ask_backoff_seconds
    with _ask_cache_lock:
        delay = _ask_backoff_seconds
        if retry_after:
            try:
                delay = max(60.0, float(retry_after))
            except ValueError:
                try:
                    delay = max(60.0, (parsedate_to_datetime(retry_after)
                                      - datetime.now(timezone.utc)).total_seconds())
                except (ValueError, TypeError):
                    pass
        _ask_retry_until = max(_ask_retry_until, time.monotonic() + delay)
        _ask_backoff_seconds = min(3600.0, _ask_backoff_seconds * 2)


def _read_public_source(url, deadline):
    """Only fixed official GET sources; never follow a listed URL or redirect."""
    if not (_DETAIL_RE.fullmatch(url) or re.fullmatch(
            r'https://askurl\.io/api/v1/urls/[a-fA-F0-9]{64}/summary', url)):
        raise ValueError('허용되지 않은 분석 자료 주소')
    if _ask_retry_seconds() > 0:
        raise AskRateLimited('ASK URL 요청 제한 대기 중')
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('ASK URL 준비 시간 제한')
    # Nurilab's .html links redirect to the same extensionless article URL.
    # Request that observed canonical form directly; never follow arbitrary redirects.
    request_url = url[:-5] if _DETAIL_RE.fullmatch(url) else url
    req = Request(request_url, headers={'User-Agent': 'KoBERT-PhishingResearch/1.0',
                                'Accept': 'application/json,text/html'}, method='GET')
    try:
        with build_opener(NoRedirect()).open(req, timeout=min(ASKURL_TIMEOUT, remaining)) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
    except HTTPError as exc:
        if exc.code == 429:
            _note_rate_limit(exc.headers.get('Retry-After') if exc.headers else None)
            raise AskRateLimited('공개 분석 서비스 요청 제한') from exc
        raise
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError('분석 자료 크기 초과')
    if time.monotonic() > deadline:
        raise TimeoutError('ASK URL 준비 시간 제한')
    return raw.decode('utf-8-sig')


def _cached_public_source(url, deadline):
    now = time.monotonic()
    with _ask_cache_lock:
        entry = _ask_cache.get(url)
        if entry and now < entry[0]:
            return entry[1]
    text = _read_public_source(url, deadline)
    with _ask_cache_lock:
        for key in [k for k, value in _ask_cache.items() if time.monotonic() >= value[0]]:
            del _ask_cache[key]
        if len(_ask_cache) >= 2048:
            del _ask_cache[next(iter(_ask_cache))]
        _ask_cache[url] = (time.monotonic() + REFRESH_SECONDS, text)
    return text


def _destination_record(item, summary, analysis_url):
    if not isinstance(summary, dict):
        raise ValueError('ASK URL 결과 형식 오류')
    if canonical_url(summary.get('url')) != item['url']:
        raise ValueError('ASK URL 원래 주소 불일치')
    if summary.get('classification') not in ('phishing', 'fraud', 'scam'):
        return None  # Do not inherit a threat verdict from an unconfirmed/benign result.
    final_url = canonical_url(summary.get('final_url'))
    if not final_url:
        raise ValueError('ASK URL 최종 주소 누락')
    redirects = summary.get('url_redirects')
    if redirects:
        if (not isinstance(redirects, list) or len(redirects) > 100
                or canonical_url(redirects[0]) != item['url']
                or canonical_url(redirects[-1]) != final_url):
            raise ValueError('ASK URL 연결 기록 불일치')
    stamp = summary.get('scan_at')
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
        raise ValueError('ASK URL 분석 시각 누락')
    scanned = datetime.fromtimestamp(stamp / 1000, KST)
    if scanned.year < 2000 or scanned > datetime.now(KST) + timedelta(minutes=5):
        raise ValueError('ASK URL 분석 시각 오류')
    if final_url == item['url']:
        return None
    return {**item, 'url': final_url, 'registered_url': item['url'],
            'match_type': 'askurl_final', 'analysis_url': analysis_url,
            'scan_at': scanned.isoformat(timespec='seconds')}


def _resolve_destination(item, deadline):
    detail = item.get('detail_url')
    if not detail:
        return None
    parser = _AskLinks()
    parser.feed(_cached_public_source(detail, deadline))
    if not parser.links:
        with _ask_cache_lock:
            _ask_cache.pop(detail, None)
        raise ValueError('ASK URL 공개 결과 링크 없음')
    last_error = None
    for link in parser.links:
        identifier = _ASK_RESULT_RE.fullmatch(link).group(1)
        endpoint = f'https://askurl.io/api/v1/urls/{identifier}/summary'
        try:
            summary = json.loads(_cached_public_source(endpoint, deadline))
            return _destination_record(item, summary, link)
        except (ValueError, OSError, OverflowError, TypeError) as exc:
            with _ask_cache_lock:
                _ask_cache.pop(endpoint, None)
            last_error = exc
    raise last_error or ValueError('ASK URL 결과 확인 불가')


def _enrich_destinations(records):
    merged = dict(records)
    candidates = sorted((item for item in records.values() if item.get('detail_url')),
                        key=lambda item: urlsplit(item['url']).hostname not in _SHORT_LINK_HOSTS)
    if not candidates:
        return merged, {'registered': len(records), 'destinations': 0, 'checked': 0, 'pending': 0}
    deadline = time.monotonic() + ASKURL_BUDGET
    checked = 0
    errors = []
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='askurl-public')
    futures = {executor.submit(_resolve_destination, item, deadline): item for item in candidates}
    try:
        for future in as_completed(futures, timeout=ASKURL_BUDGET):
            try:
                entry = future.result()
                checked += 1
                if entry:
                    # A directly listed URL always takes precedence over an inferred alias.
                    current = merged.get(entry['url'])
                    if current is None or (current.get('match_type') == 'askurl_final'
                                           and entry['scan_at'] > current['scan_at']):
                        merged[entry['url']] = entry
            except Exception as exc:
                errors.append(type(exc).__name__)
    except FuturesTimeout:
        errors.append('TimeBudgetExceeded')
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    status = {'registered': len(records), 'destinations': len(merged) - len(records),
              'checked': checked, 'pending': len(candidates) - checked,
              'errors': sorted(set(errors)), 'retry_after_seconds': round(_ask_retry_seconds())}
    return merged, status


def preparation_status():
    """Read-only diagnostics; no network access on request paths."""
    with _lock:
        return dict(_last_refresh_status)


def window_days():
    try:
        days = int(os.getenv('KOBERT_NURILAB_DAYS', '7'))
        return days if 5 <= days <= 7 else 7
    except ValueError:
        return 7


def korea_today():
    return datetime.now(KST).date()


def published_date(value):
    match = re.fullmatch(r'\s*(\d{4})[.\-/]\s*(\d{1,2})[.\-/]\s*(\d{1,2})\s*', value)
    if match:
        try:
            return date(*map(int, match.groups()))
        except ValueError:
            pass
    return None


def _recent(item, days, today):
    day = published_date(item['published_date'])
    return day is not None and today - timedelta(days=days-1) <= day <= today


def fetch_page(page, deadline):
    params = urlencode({'page': page, 'per_page': 100, 'year': 'all', 'detects': ','.join(DETECTION_KINDS)})
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('누리랩 목록 조회 시간 초과')
    request = Request(ENDPOINT + '?' + params,
                      headers={'User-Agent': 'KoBERT-PhishingResearch/1.0', 'Accept': 'application/json'})
    with build_opener(NoRedirect()).open(request, timeout=min(4.0, remaining)) as response:
        raw = response.read(2 * 1024 * 1024 + 1)
    if time.monotonic() > deadline:
        raise TimeoutError('누리랩 목록 조회 시간 초과')
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError('목록 응답 크기 초과')
    result = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(result.get('items'), list) or not all(isinstance(x, str) for x in result['items']):
        raise ValueError('누리랩 목록 응답 구조 변경')
    if result.get('current_page') != page or not isinstance(result.get('total_pages'), int):
        raise ValueError('누리랩 페이지 정보 오류')
    return result


def collect_recent(days, today):
    records = {}
    cutoff = today - timedelta(days=days-1)
    deadline = time.monotonic() + FETCH_BUDGET
    previous = None
    for page in range(1, 11):
        result = fetch_page(page, deadline)
        if not result['items']:
            if result.get('total_items') != 0:
                raise ValueError('예상하지 못한 빈 페이지')
            return records
        items = parse_alerts(''.join(result['items']))
        if not items:
            raise ValueError('유효한 피싱/사기 항목을 찾지 못했습니다')
        dates = [published_date(item['published_date']) for item in items]
        if any(day is None for day in dates):
            raise ValueError('누리랩 게시일 형식 변경')
        ordered = ([previous] if previous else []) + dates
        if any(left < right for left, right in zip(ordered, ordered[1:])):
            raise ValueError('누리랩 최신순 정렬이 변경되었습니다')
        for item, day in zip(items, dates):
            if cutoff <= day <= today:
                records.setdefault(item['url'], item)
        if dates[-1] < cutoff or page >= result['total_pages']:
            return records
        previous = dates[-1]
    raise ValueError('누리랩 조회 페이지 상한 초과')


def refresh_cache():
    """Compatibility name; refresh process memory only, with no disk writes."""
    global _index, _expires_at, _loaded_days, _next_retry, _last_refresh_status
    # Another request is already refreshing: do not queue duplicate downloads.
    if not _refresh_lock.acquire(blocking=False):
        return None
    try:
        days = window_days()
        records = collect_recent(days, korea_today())
        # Make direct records available immediately, even while enrichment is in progress.
        with _lock:
            carry = {key: value for key, value in _index.items()
                     if time.monotonic() < _expires_at and value.get('match_type') == 'askurl_final'
                     and value.get('registered_url') in records
                     and value['published_date'] == records[value['registered_url']]['published_date']
                     and value['kind'] == records[value['registered_url']]['kind']}
            _index, _loaded_days = {**carry, **records}, days
            _expires_at = time.monotonic() + REFRESH_SECONDS + INITIAL_REFRESH_TIMEOUT
            _next_retry = 0.0
            _last_refresh_status = {'registered': len(records), 'destinations': 0,
                                    'pending': sum(bool(x.get('detail_url')) for x in records.values())}
        enriched, status = _enrich_destinations(records)
        with _lock:
            _index = enriched
            _last_refresh_status = status
        if status['registered']:
            print(f"  [누리랩 + ASK URL] 등록 {status['registered']}개, 추가 목적지 {status['destinations']}개, 확인 미완료 {status['pending']}개")
            if status.get('retry_after_seconds'):
                print(f"  [ASK URL] 요청 제한: 약 {status['retry_after_seconds']}초 후 재시도. 준비된 목록으로 검사합니다.")
        return len(enriched)
    except Exception:
        with _lock:
            _index, _expires_at = {}, 0.0
            _next_retry = time.monotonic() + 60.0
            _last_refresh_status = {'error': 'feed_unavailable'}
        raise
    finally:
        _refresh_lock.release()


def _refresh_worker(stop_event, first_attempt=None):
    """Initial fetch, then hourly refresh; failed fetches retry after a minute."""
    while not stop_event.is_set():
        if os.getenv('KOBERT_NURILAB_ENABLED', '1') != '1':
            delay = 60.0
        else:
            try:
                count = refresh_cache()
                if count is not None:
                    print(f'  [누리랩] 최근 {window_days()}일 목록 {count}개 준비 완료 (1시간마다 갱신)')
                delay = (max(60.0, _ask_retry_seconds()) if preparation_status().get('pending', 0)
                         else REFRESH_SECONDS) if count is not None else 1.0
            except Exception as exc:
                print(f'  [누리랩] 목록 갱신 실패, 60초 후 재시도: {type(exc).__name__}')
                delay = 60.0
        if first_attempt is not None:
            first_attempt.set()
        if stop_event.wait(delay):
            break


def start_background_refresh():
    """Idempotent, nonblocking startup hook, once per server process."""
    global _worker, _stop_event, _first_attempt
    if os.getenv('KOBERT_NURILAB_ENABLED', '1') != '1':
        return False
    with _worker_lock:
        if _worker is not None and _worker.is_alive():
            return False
        _stop_event = threading.Event()
        _first_attempt = threading.Event()
        _worker = threading.Thread(target=_refresh_worker, args=(_stop_event, _first_attempt),
                                   daemon=True, name='nurilab-refresh')
        _worker.start()
        return True


def stop_background_refresh(timeout=2.0):
    """Optional server shutdown hook; it does not start or cancel inference."""
    with _worker_lock:
        worker = _worker
        if _stop_event is not None:
            _stop_event.set()
    if worker is not None and worker is not threading.current_thread():
        worker.join(timeout=timeout)


def wait_for_initial_refresh(timeout=INITIAL_REFRESH_TIMEOUT):
    """Startup-only bounded wait. Never call this in URL lookup/inference."""
    with _worker_lock:
        event = _first_attempt
    if os.getenv('KOBERT_NURILAB_ENABLED', '1') != '1' or event is None:
        return False
    return event.wait(timeout)


def lookup(url):
    if os.getenv('KOBERT_NURILAB_ENABLED', '1') != '1':
        return None
    key = canonical_url(url)
    if not key:
        return None
    days = window_days()
    # Memory lookup only: no downloads, thread creation or waits in this path.
    with _lock:
        if _loaded_days != days or time.monotonic() >= _expires_at:
            return None
        item = _index.get(key)
        return dict(item) if item and _recent(item, days, korea_today()) else None


def report_for(url):
    item = lookup(url)
    if item is None:
        return None
    is_fraud = item.get('kind') == '사기'
    warning = ('사기 사이트로 등록된 이력이 있는 사이트입니다.' if is_fraud else
               '해당 사이트는 이미 기록되어있는 피싱 사이트입니다. 접속을 삼가해주세요.')
    trigger = f"누리랩 피싱주의보{' 사기' if is_fraud else ''} URL 일치 (게시일: {item['published_date']}, 출처: {SOURCE})"
    if item.get('match_type') == 'askurl_final':
        warning += ' 누리랩 등록 링크의 ASK URL 분석 기록에서 확인된 연결 목적지입니다.'
        trigger = (f"누리랩 등록 링크의 최종 목적지 일치 (게시일: {item['published_date']}, "
                   f"등록 주소: {item['registered_url']}, 분석 시각: {item['scan_at']}, "
                   f"출처: {item['analysis_url']}; 현재 연결 상태를 확인한 결과는 아님)")
    # Same fields, nesting and judgment vocabulary as the existing main report.
    return {
        'url': url, 'judgment': 'unnormal', 'riskLevel': 'HIGH', 'threat_score': 99.0,
        'threat_type': '사기 사이트' if is_fraud else '일반 기관/기업 사칭 피싱', 'site_category': '일반',
        'evidence': {
            'heuristic_evidence': {
                'detected_actions': ['요구 정보 없음'],
                'rule_trigger': trigger
            },
            'ai_semantic_evidence': {
                'suspect_sentence': item['title'],
                'ai_inference_logic': warning
            }
        }
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true', required=True)
    parser.parse_args()
    print(f'최근 {window_days()}일 피싱/사기 URL {refresh_cache()}개 조회 (메모리 전용, 파일 저장 없음)')
