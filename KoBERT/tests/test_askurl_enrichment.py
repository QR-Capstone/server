"""Offline checks for official-source enrichment; no threat URL is contacted."""
import contextlib
from datetime import date
import io
import json
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nurilab_feed as feed

ID = 'e8947247c642247e3da6cef59ade314fa630d448da7f51b46fe00b9f19126b0d'
ANALYSIS = f'https://askurl.io/urls/{ID}'
API = f'https://askurl.io/api/v1/urls/{ID}/summary'
DETAIL = ('https://www.nurilab.com/kr/alert/phishing_alert/phishing_alert_paper/2026/'
          'phishing_alert_paper20261118_17949830976081_7151.html')
ORIGINAL = 'https://sw.run/KWO7f'
FINAL = 'https://vantai.cc/kr/'
ITEM = {'url': ORIGINAL, 'title': '경찰청 사칭 피싱 사이트 주의', 'kind': '피싱',
        'published_date': '2026. 11. 18', 'detail_url': DETAIL}
SUMMARY = {'url': ORIGINAL, 'classification': 'phishing', 'final_url': FINAL,
           'url_redirects': [ORIGINAL, 'http://vantai.cc/kr', 'https://vantai.cc/kr', FINAL],
           'scan_at': 1794919363193}
ROW = '''<div class="page_content" data-detect="피싱">
<div class="content_title_item"><a class="load-content-phishing"
data-path="2026" data-file="phishing_alert_paper20261118_17949830976081_7151.html">경찰청 사칭 피싱 사이트 주의</a></div>
<div class="content_target_item" title="https://sw[.]run/KWO7f">https://sw[.]run/KWO7f</div>
<div class="content_date_item">2026. 11. 18</div></div>'''


class AskEnrichmentTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(feed.os.environ, {'KOBERT_NURILAB_ENABLED': '1', 'KOBERT_NURILAB_DAYS': '7'}))
        self.stack.enter_context(patch.object(feed, 'korea_today', return_value=date(2026, 11, 24)))
        self.stack.enter_context(patch.multiple(feed, _index={}, _loaded_days=None, _expires_at=0.,
                                              _ask_cache={}, _last_refresh_status={},
                                              _ask_retry_until=0., _ask_backoff_seconds=300.))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def source(self, url, deadline):
        if url == DETAIL:
            return f'<a href="{ANALYSIS}">https://sw[.]run/KWO7f</a>'
        if url == API:
            return json.dumps(SUMMARY)
        self.fail('Unexpected network destination: ' + url)

    def refresh(self):
        with patch.object(feed, 'collect_recent', return_value={ORIGINAL: dict(ITEM)}), \
             patch.object(feed, '_read_public_source', side_effect=self.source) as fetch:
            count = feed.refresh_cache()
        return count, fetch

    def test_parse_detail_link_without_requesting_threat_url(self):
        self.assertEqual(feed.parse_alerts(ROW), [ITEM])
        bad = ROW.replace('data-path="2026"', 'data-path="../../evil"')
        self.assertNotIn('detail_url', feed.parse_alerts(bad)[0])

    def test_resolve_only_official_sources(self):
        with patch.object(feed, '_read_public_source', side_effect=self.source) as fetch:
            entry = feed._resolve_destination(ITEM, time.monotonic() + 10)
        self.assertEqual([c.args[0] for c in fetch.call_args_list], [DETAIL, API])
        self.assertEqual(entry['url'], FINAL)
        self.assertEqual(entry['registered_url'], ORIGINAL)
        self.assertEqual(entry['scan_at'], '2026-11-17T21:42:43+09:00')

    def test_rejects_wrong_original_unconfirmed_and_bad_time(self):
        for changes in [{'url': 'https://unrelated.example/'}, {'scan_at': None},
                        {'final_url': 'javascript:alert(1)'}, {'scan_at': float('inf')},
                        {'url_redirects': [ORIGINAL, 'https://unrelated.example/']}]:
            with self.subTest(changes=changes), self.assertRaises((ValueError, OverflowError)):
                feed._destination_record(ITEM, {**SUMMARY, **changes}, ANALYSIS)
        self.assertIsNone(feed._destination_record(ITEM, {**SUMMARY, 'classification': 'normal'}, ANALYSIS))

    def test_no_alias_for_identical_final(self):
        self.assertIsNone(feed._destination_record(ITEM, {**SUMMARY, 'final_url': ORIGINAL,
                                                        'url_redirects': [ORIGINAL]}, ANALYSIS))

    def test_link_parser_rejects_foreign_hosts_and_extra_paths(self):
        parser = feed._AskLinks()
        for url in ['https://askurl.io.evil.example/urls/' + ID,
                    ANALYSIS + '?redirect=https://evil.example', ORIGINAL, 'http://askurl.io/urls/' + ID]:
            parser.feed(f'<a href="{url}">result</a>')
        self.assertEqual(parser.links, [])

    def test_request_allowlist_rejects_threat_and_status_endpoints(self):
        with patch.object(feed, 'build_opener') as opener:
            for url in [ORIGINAL, FINAL, 'https://askurl.io/api/v1/urls/status',
                        API + '?scan=true', 'https://askurl.io.evil.example/api/v1/urls/' + ID + '/summary']:
                with self.assertRaises(ValueError):
                    feed._read_public_source(url, time.monotonic() + 1)
        opener.assert_not_called()

    def test_official_article_uses_canonical_get_without_redirects(self):
        from unittest.mock import MagicMock
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value.read.return_value = b'<html/>'
        with patch.object(feed, 'build_opener', return_value=opener):
            feed._read_public_source(DETAIL, time.monotonic()+10)
        req = opener.open.call_args.args[0]
        self.assertEqual(req.full_url, DETAIL[:-5])
        self.assertEqual(req.get_method(), 'GET')

    def test_rate_limit_honors_retry_after_and_stops_new_requests(self):
        from email.message import Message
        from urllib.error import HTTPError
        from unittest.mock import MagicMock
        headers = Message()
        headers['Retry-After'] = '120'
        opener = MagicMock()
        opener.open.side_effect = HTTPError(API, 429, 'limit', headers, None)
        with patch.object(feed, 'build_opener', return_value=opener):
            with self.assertRaises(feed.AskRateLimited):
                feed._read_public_source(API, time.monotonic()+10)
            with self.assertRaises(feed.AskRateLimited):
                feed._read_public_source(DETAIL, time.monotonic()+10)
        self.assertEqual(opener.open.call_count, 1)
        self.assertGreater(feed._ask_retry_seconds(), 119)

    def test_rate_limit_defaults_to_backoff(self):
        feed._note_rate_limit(None)
        self.assertGreater(feed._ask_retry_seconds(), 299)
        feed._note_rate_limit(None)
        self.assertGreater(feed._ask_retry_seconds(), 599)

    def test_registered_short_link_is_prioritized(self):
        other = {**ITEM, 'url': 'https://other.example/'}
        with patch.object(feed, '_resolve_destination', return_value=None) as resolve:
            feed._enrich_destinations({other['url']: other, ORIGINAL: ITEM})
        self.assertEqual(resolve.call_args_list[0].args[0]['url'], ORIGINAL)

    def test_memory_exact_lookup_and_existing_report_schema(self):
        count, _ = self.refresh()
        self.assertEqual(count, 2)
        with patch.object(feed, '_read_public_source', side_effect=AssertionError('No request-time network')):
            report = feed.report_for(FINAL)
            self.assertIsNotNone(feed.lookup(ORIGINAL))
            for url in ['https://vantai.cc/', 'https://vantai.cc/kr/fine-info.html', 'http://vantai.cc/kr']:
                self.assertIsNone(feed.lookup(url))
        self.assertEqual(set(report), {'url', 'judgment', 'riskLevel', 'threat_score', 'threat_type', 'site_category', 'evidence'})
        self.assertIn('연결 목적지', report['evidence']['ai_semantic_evidence']['ai_inference_logic'])
        self.assertIn(ANALYSIS, report['evidence']['heuristic_evidence']['rule_trigger'])

    def test_alias_expires_with_publication_window(self):
        self.refresh()
        with patch.object(feed, 'korea_today', return_value=date(2026, 11, 25)):
            self.assertIsNone(feed.lookup(FINAL))
            self.assertIsNone(feed.lookup(ORIGINAL))

    def test_failure_retains_direct_record(self):
        with patch.object(feed, 'collect_recent', return_value={ORIGINAL: dict(ITEM)}), \
             patch.object(feed, '_read_public_source', side_effect=OSError('429')):
            self.assertEqual(feed.refresh_cache(), 1)
        self.assertIsNotNone(feed.lookup(ORIGINAL))
        self.assertIsNone(feed.lookup(FINAL))
        self.assertEqual(feed.preparation_status()['pending'], 1)

    def test_successful_sources_reused_on_retry(self):
        self.refresh()
        with patch.object(feed, 'collect_recent', return_value={ORIGINAL: dict(ITEM)}), \
             patch.object(feed, '_read_public_source', side_effect=AssertionError('Already cached')):
            self.assertEqual(feed.refresh_cache(), 2)

    def test_new_refresh_removes_previous_alias(self):
        self.refresh()
        with patch.object(feed, 'collect_recent', return_value={}):
            feed.refresh_cache()
        self.assertIsNone(feed.lookup(FINAL))

    def test_direct_listing_wins_over_alias(self):
        direct = {**ITEM, 'url': FINAL, 'detail_url': ''}
        with patch.object(feed, '_resolve_destination', return_value=feed._destination_record(ITEM, SUMMARY, ANALYSIS)):
            merged, _ = feed._enrich_destinations({ORIGINAL: ITEM, FINAL: direct})
        self.assertEqual(merged[FINAL], direct)

    def test_budget_does_not_wait_for_unfinished_future(self):
        release = threading.Event()
        def slow(*args):
            release.wait(2)
            return None
        try:
            with patch.object(feed, 'ASKURL_BUDGET', .02), patch.object(feed, '_resolve_destination', side_effect=slow):
                start = time.monotonic()
                merged, status = feed._enrich_destinations({ORIGINAL: ITEM})
                self.assertLess(time.monotonic() - start, .5)
                self.assertEqual(merged, {ORIGINAL: ITEM})
                self.assertEqual(status['pending'], 1)
        finally:
            release.set()


class AliasEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kobert_test_support import LocalTester
        with contextlib.redirect_stderr(io.StringIO()):
            cls.tester = LocalTester(enable_feed=False)

    def test_destination_keeps_100_when_body_is_unavailable(self):
        engine = self.tester.engine
        alias = feed._destination_record(ITEM, SUMMARY, ANALYSIS)
        with patch.multiple(feed, _index={FINAL: alias}, _loaded_days=7, _expires_at=time.monotonic()+60), \
             patch.object(feed, 'korea_today', return_value=date(2026, 11, 24)), \
             patch.dict(feed.os.environ, {'KOBERT_NURILAB_ENABLED': '1', 'KOBERT_RESULT_CACHE_ENABLED': '0'}), \
             patch.object(engine, 'extract_with_requests_and_raw_html', return_value=('Page not available', '<p/>')), \
             patch.object(engine.model, 'forward', side_effect=AssertionError('No body, no model score')), \
             contextlib.redirect_stderr(io.StringIO()):
            result = self.tester.url(FINAL, browser=False)
        self.assertEqual(result['judgment'], 'unnormal')
        self.assertEqual(result['threat_score'], 100.0)
        reason = result['evidence']['ai_semantic_evidence']['ai_inference_logic']
        self.assertIn('연결 목적지', reason)
        self.assertIn(engine.ACCESS_UNAVAILABLE_MESSAGE, reason)
        self.assertIn(ORIGINAL, result['evidence']['heuristic_evidence']['rule_trigger'])

    def test_destination_still_runs_independent_model_for_real_text(self):
        engine = self.tester.engine
        alias = feed._destination_record(ITEM, SUMMARY, ANALYSIS)
        with patch.multiple(feed, _index={FINAL: alias}, _loaded_days=7, _expires_at=time.monotonic()+60), \
             patch.object(feed, 'korea_today', return_value=date(2026, 11, 24)), \
             patch.dict(feed.os.environ, {'KOBERT_NURILAB_ENABLED': '1', 'KOBERT_RESULT_CACHE_ENABLED': '0'}), \
             patch.object(engine, 'extract_with_requests_and_raw_html',
                          return_value=('박물관 전시 일정과 관람 시간을 안내합니다.', '<p/>')), \
             patch.object(engine.model, 'forward', wraps=engine.model.forward) as forward, \
             contextlib.redirect_stderr(io.StringIO()):
            result = self.tester.url(FINAL, browser=False)
        forward.assert_called_once()
        self.assertEqual(result['threat_score'], 100.0)
        self.assertIn('KoBERT 독립 문맥 분석', result['evidence']['ai_semantic_evidence']['ai_inference_logic'])


if __name__ == '__main__':
    unittest.main()
