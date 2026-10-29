import json
from datetime import date
import time
import unittest
from unittest.mock import patch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nurilab_feed as feed


def row(url, kind='피싱', title='기관 사칭 피싱 주의'):
    return f'''<div class="page_content" data-detect="{kind}">
      <div class="content_title_item"><a>{title}</a><img src="unused"></div>
      <div class="content_target_item" title="{url}">잘린 주소...</div>
      <div class="content_date_item">2026. 10. 06</div></div>'''


class FeedTests(unittest.TestCase):
    def setUp(self):
        self.today_patch = patch.object(feed, 'korea_today', return_value=date(2026, 10, 6))
        self.today_patch.start()
        self.background_patch = patch.object(feed, 'start_background_refresh', return_value=False)
        self.background_patch.start()
        self.env = patch.dict(feed.os.environ, {'KOBERT_NURILAB_ENABLED': '1', 'KOBERT_NURILAB_DAYS': '7'})
        self.env.start()
        feed._index, feed._loaded_days, feed._expires_at, feed._next_retry = {}, None, 0, float('inf')

    def tearDown(self):
        self.env.stop()
        self.background_patch.stop()
        self.today_patch.stop()
        feed._index, feed._loaded_days, feed._expires_at, feed._next_retry = {}, None, 0, 0

    def save(self, url='https://phish.example/login', age=0):
        entry = {'url': url, 'kind': '피싱', 'title': '기관 사칭 주의', 'published_date': '2026. 10. 06'}
        feed._index = {url: entry}
        feed._loaded_days = feed.window_days()
        feed._expires_at = time.monotonic() + feed.REFRESH_SECONDS - age

    def test_full_attribute_not_truncated_text(self):
        result = feed.parse_alerts(row('https://phish[.]example/login?a=1&amp;b=2'))
        self.assertEqual(result[0]['url'], 'https://phish.example/login?a=1&b=2')
        self.assertEqual(result[0]['title'], '기관 사칭 피싱 주의')

    def test_only_phishing_not_fraud(self):
        self.assertEqual(len(feed.parse_alerts(row('https://a[.]example/') + row('https://b[.]example/', '사기'))), 1)

    def test_bad_shape_and_incomplete_url(self):
        with self.assertRaises(ValueError):
            feed.parse_alerts('<html>blocked</html>')
        self.assertEqual(feed.parse_alerts(row('https://phish[.]example/...')), [])

    def test_normalization(self):
        self.assertEqual(feed.canonical_url('hxxps://EXAMPLE[.]com:443'), 'https://example.com/')
        self.assertEqual(feed.canonical_url('https://인증활성[.]xyz/'), 'https://' + '인증활성'.encode('idna').decode() + '.xyz/')
        for value in ['javascript:alert(1)', 'https://a.example@b.example/', 'https://a.example:bad', 'https://a.example/…']:
            self.assertIsNone(feed.canonical_url(value))

    def test_exact_scope(self):
        self.save()
        self.assertIsNotNone(feed.lookup('https://PHISH.example:443/login'))
        for value in ['https://phish.example/', 'https://phish.example/login2',
                      'https://phish.example/login?new=1', 'http://phish.example/login',
                      'https://other.phish.example/login', 'https://safe.example/?next=https://phish.example/login']:
            self.assertIsNone(feed.lookup(value))

    def test_no_shared_domain_block(self):
        self.save('https://example.pages.dev/phishing')
        self.assertIsNone(feed.lookup('https://other.pages.dev/phishing'))
        self.assertIsNone(feed.lookup('https://example.pages.dev/normal'))

    def test_missing_and_stale_are_misses(self):
        self.assertIsNone(feed.lookup('https://phish.example/login'))
        self.save(age=feed.REFRESH_SECONDS + 5)
        self.assertIsNone(feed.lookup('https://phish.example/login'))

    def test_disabled(self):
        self.save()
        with patch.dict(feed.os.environ, {'KOBERT_NURILAB_ENABLED': '0'}):
            self.assertIsNone(feed.lookup('https://phish.example/login'))

    def test_refresh_failure_clears_stale_data(self):
        self.save()
        with patch.object(feed, 'collect_recent', side_effect=OSError('offline')):
            with self.assertRaises(OSError):
                feed.refresh_cache()
        self.assertIsNone(feed.lookup('https://phish.example/login'))

    def test_refresh_replaces_not_appends(self):
        self.save()
        entries = feed.parse_alerts(row('https://new[.]example/'))
        with patch.object(feed, 'collect_recent', return_value={item['url']: item for item in entries}):
            self.assertEqual(feed.refresh_cache(), 1)
        self.assertIsNone(feed.lookup('https://phish.example/login'))
        self.assertIsNotNone(feed.lookup('https://new.example/'))

    def test_request_never_fetches_synchronously(self):
        feed._next_retry = 0
        with patch.object(feed, 'collect_recent', return_value={}) as fetch:
            feed.lookup('https://phish.example/login')
            feed.lookup('https://phish.example/login')
            self.assertEqual(fetch.call_count, 0)

    def test_date_window_boundaries(self):
        for stamp, expected in [('2026. 10. 06', True), ('2026. 09. 30', True),
                                ('2026. 09. 29', False), ('2026. 10. 07', False),
                                ('bad', False)]:
            self.save()
            feed._index['https://phish.example/login']['published_date'] = stamp
            self.assertEqual(feed.lookup('https://phish.example/login') is not None, expected)
        with patch.dict(feed.os.environ, {'KOBERT_NURILAB_DAYS': '5'}):
            self.save()
            feed._index['https://phish.example/login']['published_date'] = '2026. 10. 01'
            self.assertIsNone(feed.lookup('https://phish.example/login'))
            feed._index['https://phish.example/login']['published_date'] = '2026. 10. 02'
            self.assertIsNotNone(feed.lookup('https://phish.example/login'))

    def test_midnight_expires_old_post_even_with_fresh_memory(self):
        self.save()
        feed._index['https://phish.example/login']['published_date'] = '2026. 09. 30'
        with patch.object(feed, 'korea_today', return_value=date(2026, 10, 7)):
            self.assertIsNone(feed.lookup('https://phish.example/login'))

    def test_pagination_stops_at_old_date(self):
        first = row('https://a[.]example/').replace('2026. 10. 06', '2026. 10. 05')
        old = row('https://old[.]example/').replace('2026. 10. 06', '2026. 09. 29')
        with patch.object(feed, 'fetch_page', return_value={'items': [first, old], 'total_pages': 15}) as fetch:
            records = feed.collect_recent(7, date(2026, 10, 6))
        self.assertEqual(list(records), ['https://a.example/'])
        self.assertEqual(fetch.call_count, 1)

    def test_year_boundary(self):
        recent = row('https://a[.]example/').replace('2026. 10. 06', '2025. 12. 28')
        with patch.object(feed, 'fetch_page', return_value={'items': [recent], 'total_pages': 1}):
            self.assertEqual(len(feed.collect_recent(7, date(2026, 1, 3))), 1)

    def test_invalid_dates_and_sort_order_rejected(self):
        for rows in [[row('https://a[.]example/').replace('2026. 10. 06', 'unknown')],
                     [row('https://a[.]example/').replace('2026. 10. 06', '2026. 10. 01'), row('https://b[.]example/')]]:
            with patch.object(feed, 'fetch_page', return_value={'items': rows, 'total_pages': 2}):
                with self.assertRaises(ValueError):
                    feed.collect_recent(7, date(2026, 10, 6))


    def test_hourly_schedule_and_retry(self):
        class StopAfterTwo:
            def __init__(self): self.delays = []
            def is_set(self): return False
            def wait(self, delay):
                self.delays.append(delay)
                return len(self.delays) == 2
        for outcomes, delays in [([1, 1], [3600, 3600]),
                                 ([OSError('offline'), 1], [60, 3600])]:
            stop = StopAfterTwo()
            with patch.object(feed, 'refresh_cache', side_effect=outcomes) as refresh:
                feed._refresh_worker(stop)
            self.assertEqual(refresh.call_count, 2)
            self.assertEqual(stop.delays, delays)

    def test_lookup_does_not_start_or_wait_for_background_work(self):
        with patch.object(feed, 'start_background_refresh') as start, \
                patch.object(feed, 'wait_for_initial_refresh') as wait:
            feed.lookup('https://unknown.example/')
        start.assert_not_called()
        wait.assert_not_called()

    def test_hit_skips_initialization_network_and_model(self):
        import koBERT as engine
        self.save('https://naver.com/listed-path')
        with patch.object(engine, 'engine_initialized', False), \
                patch.object(engine, 'warmup_engine', side_effect=AssertionError('no warmup')), \
                patch.object(engine, 'extract_with_requests_and_raw_html', side_effect=AssertionError('no HTTP')), \
                patch.object(engine, 'model', side_effect=AssertionError('no inference')):
            result = engine.predict_phishing_result('https://naver.com/listed-path')
        self.assertEqual(result['judgment'], 'unnormal')
        self.assertEqual(result['riskLevel'], 'HIGH')
        self.assertEqual(set(result), {'url', 'judgment', 'riskLevel', 'threat_score',
                                      'threat_type', 'site_category', 'evidence'})
        self.assertEqual(set(result['evidence']), {'heuristic_evidence', 'ai_semantic_evidence'})
        self.assertEqual(result['evidence']['ai_semantic_evidence']['ai_inference_logic'], '해당 사이트는 이미 기록되어있는 피싱 사이트입니다. 접속을 삼가해주세요.')

    def test_child_hit_precedes_child_fetch_and_miss_runs_original_path(self):
        import contextlib, io
        from types import SimpleNamespace
        from unittest.mock import Mock
        import torch
        import koBERT as engine
        self.save()
        text = '박물관 전시 일정과 관람 시간을 안내합니다. ' * 30
        fake_tokenizer = Mock(return_value={'input_ids': torch.zeros((1, 4), dtype=torch.long),
                                            'attention_mask': torch.ones((1, 4), dtype=torch.long)})
        fake_tokenizer.tokenize.return_value = ['관람']
        fake_model = Mock(return_value=SimpleNamespace(logits=torch.tensor([[20., -20.]]),
                                                       attentions=(torch.zeros((1, 1, 4, 4)),)))
        with patch.multiple(engine, engine_initialized=True, device=torch.device('cpu'),
                            tokenizer=fake_tokenizer, model=fake_model), \
                patch.object(engine, 'extract_with_requests_and_raw_html', return_value=(text, '<html></html>')) as fetch, \
                patch.object(engine, 'extract_deep_links', return_value=['https://phish.example/login']), \
                patch.object(engine, 'extract_with_playwright_and_raw_html', side_effect=AssertionError('no browser')), \
                contextlib.redirect_stdout(io.StringIO()):
            result = engine.predict_phishing_result('https://museum.example/')
        fetch.assert_called_once_with('https://museum.example/')
        fake_model.assert_called_once()
        self.assertEqual(result['url'], 'https://phish.example/login')
        self.assertEqual(result['judgment'], 'unnormal')


class BackgroundTests(unittest.TestCase):
    def test_initial_download_is_concurrent_and_lookup_never_waits(self):
        import threading
        fetching, release = threading.Event(), threading.Event()
        def collect(*args):
            fetching.set()
            release.wait(3)
            return {}
        with patch.dict(feed.os.environ, {'KOBERT_NURILAB_ENABLED': '1'}), \
                patch.object(feed, 'collect_recent', side_effect=collect) as download:
            try:
                self.assertTrue(feed.start_background_refresh())
                self.assertTrue(fetching.wait(1))
                self.assertFalse(feed.start_background_refresh())
                self.assertFalse(feed.wait_for_initial_refresh(timeout=0))
                self.assertIsNone(feed.lookup('https://unknown.example/'))
                self.assertFalse(release.is_set())
                release.set()
                self.assertTrue(feed.wait_for_initial_refresh(timeout=1))
                self.assertEqual(download.call_count, 1)
            finally:
                release.set()
                feed.stop_background_refresh(timeout=2)
        self.assertFalse(feed._worker.is_alive())
        feed._worker = feed._stop_event = feed._first_attempt = None

    def test_model_warmup_starts_feed_and_waits_only_on_first_initialization(self):
        from unittest.mock import Mock
        from types import SimpleNamespace
        import torch
        import koBERT as engine
        tok = Mock(return_value={'input_ids': torch.zeros((1, 4), dtype=torch.long),
                                 'attention_mask': torch.ones((1, 4), dtype=torch.long)})
        model = Mock(return_value=SimpleNamespace(logits=torch.zeros((1, 2))))
        with patch.multiple(engine, engine_initialized=False, device=None, tokenizer=None, model=None), \
                patch.object(engine, 'load_runtime', return_value=(tok, model)), \
                patch.object(torch.cuda, 'is_available', return_value=False), \
                patch.object(feed, 'start_background_refresh') as start, \
                patch.object(feed, 'wait_for_initial_refresh', return_value=False) as wait:
            engine.warmup_engine(False)
            self.assertTrue(engine.engine_initialized)  # Source unavailable does not block service.
            engine.warmup_engine(False)
        start.assert_called_once()
        wait.assert_called_once()


if __name__ == '__main__':
    unittest.main()
