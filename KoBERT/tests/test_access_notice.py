"""Offline regressions: inaccessible pages must not receive model safety scores."""
import contextlib
import io
import os
import unittest
from unittest.mock import patch

from kobert_test_support import LocalTester

MESSAGE = ('접근 제한 안내가 표시되어 실제 본문을 확보하지 못했습니다. '
           'KoBERT 문맥 분석을 수행할 수 없습니다. 접근에 유의해주시길 바랍니다.')


class AccessNoticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stderr(io.StringIO()):
            cls.tester = LocalTester(enable_feed=False)
        cls.engine = cls.tester.engine

    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {'KOBERT_NURILAB_ENABLED': '0',
                                                       'KOBERT_RESULT_CACHE_ENABLED': '1'}))
        self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        self.engine._result_cache.clear()
        self.addCleanup(self.engine._result_cache.clear)

    def test_only_short_notice_matches(self):
        for value in ['Page not available', ' PAGE\nNOT AVAILABLE. ',
                      'Page not available. Page not available', 'This page is not available',
                      '접근이 제한되었습니다.']:
            with self.subTest(value=value):
                self.assertTrue(self.engine._is_access_notice(value))
        for value in ['', 'Page not available 오류 해결 방법을 안내합니다.',
                      '서비스 소개. Page not available', '404 Not Found',
                      '이 페이지는 개인정보 보호를 위한 접근 제한 안내를 제공합니다.']:
            with self.subTest(value=value):
                self.assertFalse(self.engine._is_access_notice(value))

    def test_root_notice_unknown_exact_message_no_model_no_cache(self):
        with patch.object(self.engine, 'extract_with_requests_and_raw_html',
                          return_value=('Page not available', '<p>Page not available</p>')) as fetch, \
             patch.object(self.engine.model, 'forward', side_effect=AssertionError('No inference')) as forward:
            for _ in range(2):
                result = self.tester.url('https://unlisted.example/kr/', browser=False)
                self.assertEqual(result['judgment'], 'unknown')
                self.assertEqual(result['riskLevel'], 'UNKNOWN')
                self.assertEqual(result['evidence']['ai_semantic_evidence']['ai_inference_logic'], MESSAGE)
                self.assertNotIn('threat_score', result)
            self.assertEqual(fetch.call_count, 2)
            forward.assert_not_called()
            self.assertFalse(self.engine._result_cache)

    def test_dynamic_notice_unknown(self):
        with patch.object(self.engine, 'extract_with_requests_and_raw_html', return_value=('[오류] Timeout', '')), \
             patch.object(self.engine, 'extract_with_playwright_and_raw_html',
                          return_value=('Page not available', '<p/>')) as rendered, \
             patch.object(self.engine.model, 'forward', side_effect=AssertionError('No inference')):
            result = self.tester.url('https://unlisted.example/kr/', browser=True)
        rendered.assert_called_once()
        self.assertEqual(result['judgment'], 'unknown')

    def test_listed_phishing_and_fraud_keep_policy_without_model(self):
        for kind in ['피싱', '사기']:
            with self.subTest(kind=kind), \
                 patch.object(self.engine.nurilab_feed, 'lookup', return_value={
                     'kind': kind, 'title': '오프라인 테스트', 'published_date': '2026. 11. 24'}), \
                 patch.object(self.engine, 'extract_with_requests_and_raw_html', return_value=('[오류] Timeout', '')), \
                 patch.object(self.engine, 'extract_with_playwright_and_raw_html', return_value=('Page not available', '<p/>')), \
                 patch.object(self.engine.model, 'forward', side_effect=AssertionError('No inference')):
                result = self.tester.url('https://listed.example/', browser=True)
            self.assertEqual(result['judgment'], 'unnormal')
            self.assertEqual(result['threat_score'], 100.0)
            self.assertIn(MESSAGE, result['evidence']['ai_semantic_evidence']['ai_inference_logic'])
            self.assertFalse(self.engine._result_cache)

    def test_static_valid_body_survives_listed_dynamic_notice(self):
        text = '박물관 전시 일정과 관람 시간을 안내합니다.'
        with patch.object(self.engine.nurilab_feed, 'lookup', return_value={
                  'kind': '피싱', 'title': '테스트', 'published_date': '2026. 11. 24'}), \
             patch.object(self.engine, 'extract_with_requests_and_raw_html', return_value=(text, '<p/>')), \
             patch.object(self.engine, 'extract_with_playwright_and_raw_html', return_value=('Page not available', '<p/>')), \
             patch.object(self.engine.model, 'forward', wraps=self.engine.model.forward) as forward:
            result = self.tester.url('https://listed.example/', browser=True)
        forward.assert_called_once()
        self.assertNotIn(MESSAGE, result['evidence']['ai_semantic_evidence']['ai_inference_logic'])

    def test_child_notice_does_not_fall_through_to_normal(self):
        main = 'https://museum.example/'
        child = 'https://museum.example/details'
        text = '박물관 전시 일정과 관람 시간을 안내합니다. 상세 안내는 현장에서 확인할 수 있습니다.'
        def fetch(url):
            return (text, '<p/>') if url == main else ('Page not available', '<p/>')
        with patch.object(self.engine, 'extract_with_requests_and_raw_html', side_effect=fetch), \
             patch.object(self.engine, 'extract_deep_links', return_value=[child]), \
             patch.object(self.engine.model, 'forward', wraps=self.engine.model.forward) as forward:
            result = self.tester.url(main, browser=False)
        forward.assert_called_once()  # Main page only; never classify the child notice.
        self.assertEqual(result['judgment'], 'unknown')
        self.assertEqual(result['detectedUrl'], child)
        self.assertFalse(self.engine._result_cache)


if __name__ == '__main__':
    unittest.main()
