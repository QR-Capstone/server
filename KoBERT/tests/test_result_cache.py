"""URL result cache regressions with controlled time and no network/model I/O."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import koBERT as engine


def report(url, label='normal', reason='테스트 문맥 분석'):
    return {'url': url, 'judgment': label, 'riskLevel': 'HIGH' if label == 'unnormal' else 'LOW',
            'threat_score': 100.0 if label == 'unnormal' else 3.0,
            'evidence': {'ai_semantic_evidence': {'ai_inference_logic': reason}}}


class CacheTests(unittest.TestCase):
    def setUp(self):
        engine.clear_result_cache()
        self.addCleanup(engine.clear_result_cache)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.dict(engine.os.environ, {
            'KOBERT_RESULT_CACHE_ENABLED': '1', 'KOBERT_RESULT_CACHE_HOURS': '2',
            'USE_PLAYWRIGHT_IN_ANALYZE': '0', 'KOBERT_NURILAB_ENABLED': '1'}))
        self.now = 100.0
        self.stack.enter_context(patch.object(engine.time, 'monotonic', side_effect=lambda: self.now))
        self.url = 'https://parent.example/'
        self.child = 'https://child.example/login'
        self.listings = {}
        self.children = []
        self.cacheable = True
        self.fixed_report = None
        self.precheck = self.stack.enter_context(patch.object(engine, '_nurilab_precheck',
            side_effect=lambda url: deepcopy(self.listings.get(url))))
        self.infer = self.stack.enter_context(patch.object(engine, '_predict_phishing_result_uncached',
            side_effect=self.analyze))

    def analyze(self, url, known_report, context):
        context['cacheable'] = self.cacheable
        for child in self.children:
            context['matches'].append((child, deepcopy(self.listings.get(child))))
        if self.fixed_report is not None:
            return deepcopy(self.fixed_report)
        match = known_report or next((self.listings[c] for c in self.children if c in self.listings), None)
        return deepcopy(match) if match else report(url)

    def test_hit_skips_collection_and_inference_but_rechecks_feed(self):
        first = engine.predict_phishing_result(self.url)
        second = engine.predict_phishing_result(self.url)
        self.assertEqual(first, second)
        self.infer.assert_called_once()
        self.assertEqual(self.precheck.call_count, 2)

    def test_mutating_caller_report_does_not_modify_cached_report(self):
        first = engine.predict_phishing_result(self.url)
        first['evidence']['ai_semantic_evidence']['ai_inference_logic'] = 'tampered'
        second = engine.predict_phishing_result(self.url)
        self.assertEqual(second['evidence']['ai_semantic_evidence']['ai_inference_logic'], '테스트 문맥 분석')
        second['judgment'] = 'unknown'
        self.assertEqual(engine.predict_phishing_result(self.url)['judgment'], 'normal')

    def test_fixed_expiry_does_not_slide_on_cache_hits(self):
        engine.predict_phishing_result(self.url)
        self.now += 7199
        engine.predict_phishing_result(self.url)
        self.infer.assert_called_once()
        self.now += 1
        engine.predict_phishing_result(self.url)
        self.assertEqual(self.infer.call_count, 2)

    def test_ttl_configuration_and_shortening(self):
        for value, seconds in [('1', 3600), ('2', 7200), ('3', 10800), ('1.5', 5400),
                               ('0', 7200), ('4', 7200), ('bad', 7200), ('nan', 7200), ('inf', 7200)]:
            with patch.dict(engine.os.environ, {'KOBERT_RESULT_CACHE_HOURS': value}):
                self.assertEqual(engine._result_cache_ttl(), seconds)
        engine.predict_phishing_result(self.url)
        self.now += 3600
        with patch.dict(engine.os.environ, {'KOBERT_RESULT_CACHE_HOURS': '1'}):
            engine.predict_phishing_result(self.url)
        self.assertEqual(self.infer.call_count, 2)

    def test_new_root_listing_overrides_cached_normal(self):
        engine.predict_phishing_result(self.url)
        self.listings[self.url] = report(self.url, 'unnormal', '목록 일치')
        result = engine.predict_phishing_result(self.url)
        self.assertEqual(result['judgment'], 'unnormal')
        self.assertEqual(self.infer.call_count, 2)

    def test_listed_result_reused_then_invalidated_when_listing_expires(self):
        self.listings[self.url] = report(self.url, 'unnormal', '목록 일치')
        engine.predict_phishing_result(self.url)
        engine.predict_phishing_result(self.url)
        self.infer.assert_called_once()
        self.listings.clear()
        self.assertEqual(engine.predict_phishing_result(self.url)['judgment'], 'normal')
        self.assertEqual(self.infer.call_count, 2)

    def test_new_child_listing_overrides_cached_normal(self):
        self.children = [self.child]
        engine.predict_phishing_result(self.url)
        self.listings[self.child] = report(self.child, 'unnormal', '하위 목록 일치')
        result = engine.predict_phishing_result(self.url)
        self.assertEqual(result['url'], self.child)
        self.assertEqual(result['judgment'], 'unnormal')
        self.assertEqual(self.infer.call_count, 2)

    def test_listing_type_change_replaces_cached_warning(self):
        self.listings[self.url] = report(self.url, 'unnormal', '피싱 등록 안내')
        engine.predict_phishing_result(self.url)
        self.listings[self.url] = report(self.url, 'unnormal', '사기 사이트로 등록된 이력이 있는 사이트입니다.')
        result = engine.predict_phishing_result(self.url)
        self.assertEqual(result['evidence']['ai_semantic_evidence']['ai_inference_logic'],
                         '사기 사이트로 등록된 이력이 있는 사이트입니다.')
        self.assertEqual(self.infer.call_count, 2)

    def test_changed_or_removed_child_listing_invalidates_parent(self):
        self.children = [self.child]
        self.listings[self.child] = report(self.child, 'unnormal', '이전 게시일')
        engine.predict_phishing_result(self.url)
        self.listings[self.child] = report(self.child, 'unnormal', '변경 게시일')
        engine.predict_phishing_result(self.url)
        self.listings.clear()
        self.assertEqual(engine.predict_phishing_result(self.url)['judgment'], 'normal')
        self.assertEqual(self.infer.call_count, 3)

    def test_failures_and_incomplete_child_scans_are_not_cached(self):
        for result in (report(self.url, 'unknown'),
                       report(self.url, 'unnormal', 'KoBERT 추가 분석 불가: 본문 없음'),
                       dict(report(self.url), error='failure')):
            self.fixed_report = result
            self.infer.reset_mock()
            engine.predict_phishing_result(self.url)
            engine.predict_phishing_result(self.url)
            self.assertEqual(self.infer.call_count, 2)
            self.assertEqual(len(engine._result_cache), 0)
        self.fixed_report = None
        self.cacheable = False
        engine.predict_phishing_result(self.url)
        self.assertEqual(len(engine._result_cache), 0)

    def test_browser_settings_and_model_identity_separate_entries(self):
        engine.predict_phishing_result(self.url)
        with patch.dict(engine.os.environ, {'USE_PLAYWRIGHT_IN_ANALYZE': '1'}):
            engine.predict_phishing_result(self.url)
        with patch.object(engine, 'model', object()):
            engine.predict_phishing_result(self.url)
        self.assertEqual(self.infer.call_count, 3)

    def test_different_paths_queries_schemes_do_not_share_results(self):
        urls = ['https://a.example/login', 'https://a.example/login?x=1',
                'https://a.example/login?x=2', 'http://a.example/login',
                'https://a.example/login2', 'https://a.example/login#form']
        for url in urls:
            self.assertEqual(engine.predict_phishing_result(url)['url'], url)
        self.assertEqual(self.infer.call_count, len(urls))

    def test_disabled_and_explicit_clear(self):
        engine.predict_phishing_result(self.url)
        with patch.dict(engine.os.environ, {'KOBERT_RESULT_CACHE_ENABLED': '0'}):
            engine.predict_phishing_result(self.url)
            engine.predict_phishing_result(self.url)
        self.assertEqual(self.infer.call_count, 3)
        engine.clear_result_cache()
        engine.predict_phishing_result(self.url)
        self.assertEqual(self.infer.call_count, 4)

    def test_lru_capacity_and_expired_entry_cleanup(self):
        with patch.object(engine, '_RESULT_CACHE_LIMIT', 2):
            engine.predict_phishing_result('https://one.example/')
            engine.predict_phishing_result('https://two.example/')
            engine.predict_phishing_result('https://one.example/')
            engine.predict_phishing_result('https://three.example/')
            self.assertEqual(len(engine._result_cache), 2)
            self.assertEqual({k[0] for k in engine._result_cache}, {'https://one.example/', 'https://three.example/'})
            self.now += 7200
            engine.predict_phishing_result('https://four.example/')
            self.assertEqual(len(engine._result_cache), 1)

    def test_local_file_requests_are_not_cached(self):
        for value in ('file:///C:/page.html', 'D:\\page.html'):
            self.assertIsNone(engine._result_cache_key(value))


if __name__ == '__main__':
    unittest.main()
