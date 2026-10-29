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
from urllib.parse import urlsplit, urlunsplit, urlencode
from urllib.request import Request, build_opener, HTTPRedirectHandler

SOURCE = 'https://www.nurilab.com/kr/alert/phishing_alert'
ENDPOINT = 'https://www.nurilab.com/kr/alert/paginate_alerts'
KST = timezone(timedelta(hours=9))
REFRESH_SECONDS = 3600
FETCH_BUDGET = 12.0
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
        if row['kind'].strip() != '피싱':
            continue
        key = canonical_url(row['url'])
        if key:
            entries.append({'url': key, 'title': ' '.join(row['title'].split()),
                            'published_date': ' '.join(row['date'].split()), 'kind': '피싱'})
    return entries


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('목록 요청의 리다이렉트는 허용하지 않습니다')


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
    params = urlencode({'page': page, 'per_page': 100, 'year': 'all', 'detects': '피싱'})
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
            raise ValueError('유효한 피싱 항목을 찾지 못했습니다')
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
    global _index, _expires_at, _loaded_days, _next_retry
    # Another request is already refreshing: do not queue duplicate downloads.
    if not _refresh_lock.acquire(blocking=False):
        return None
    try:
        days = window_days()
        records = collect_recent(days, korea_today())
        with _lock:
            _index, _loaded_days = records, days
            _expires_at = time.monotonic() + REFRESH_SECONDS + FETCH_BUDGET
            _next_retry = 0.0
        return len(records)
    except Exception:
        with _lock:
            _index, _expires_at = {}, 0.0
            _next_retry = time.monotonic() + 60.0
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
                delay = REFRESH_SECONDS if count is not None else 1.0
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


def wait_for_initial_refresh(timeout=FETCH_BUDGET + 1.0):
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
    # Same fields, nesting and judgment vocabulary as the existing main report.
    return {
        'url': url, 'judgment': 'unnormal', 'riskLevel': 'HIGH', 'threat_score': 99.0,
        'threat_type': '일반 기관/기업 사칭 피싱', 'site_category': '일반',
        'evidence': {
            'heuristic_evidence': {
                'detected_actions': ['요구 정보 없음'],
                'rule_trigger': f"누리랩 피싱주의보 URL 일치 (게시일: {item['published_date']}, 출처: {SOURCE})"
            },
            'ai_semantic_evidence': {
                'suspect_sentence': item['title'],
                'ai_inference_logic': '해당 사이트는 이미 기록되어있는 피싱 사이트입니다. 접속을 삼가해주세요.'
            }
        }
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true', required=True)
    parser.parse_args()
    print(f'최근 {window_days()}일 피싱 URL {refresh_cache()}개 조회 (메모리 전용, 파일 저장 없음)')
