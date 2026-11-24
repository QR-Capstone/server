"""Offline regression tests: no URLs are contacted and no weights are loaded."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import koBERT as engine


class ListedInferenceTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.dict(engine.os.environ, {
            'USE_PLAYWRIGHT_IN_ANALYZE': '0', 'KOBERT_MAX_LEN': '512',
            'KOBERT_RESULT_CACHE_ENABLED': '0'}))
        self.url = 'https://naver.com/listed-path'
        self.text = '이용자 확인을 위해 계좌번호와 인증번호를 제출해주세요. ' * 20
        self.known = {
            'url': self.url, 'judgment': 'unnormal', 'riskLevel': 'HIGH',
            'threat_score': 100.0, 'threat_type': '일반 기관/기업 사칭 피싱',
            'site_category': '일반', 'evidence': {
                'heuristic_evidence': {'detected_actions': ['요구 정보 없음'],
                                      'rule_trigger': '누리랩 URL 일치, 게시일 및 출처'},
                'ai_semantic_evidence': {'suspect_sentence': '게시물 제목',
                                        'ai_inference_logic': '해당 사이트는 이미 기록되어있는 피싱 사이트입니다. 접속을 삼가해주세요.'}}}
        self.original = deepcopy(self.known)
        self.tok = Mock(return_value={
            'input_ids': torch.zeros((1, 8), dtype=torch.long),
            'attention_mask': torch.ones((1, 8), dtype=torch.long)})
        self.model = Mock(return_value=SimpleNamespace(logits=torch.tensor([[0., 2.]])))
        self.stack.enter_context(patch.multiple(engine, engine_initialized=True,
            device=torch.device('cpu'), tokenizer=self.tok, model=self.model))
        self.precheck = self.stack.enter_context(patch.object(engine, '_nurilab_precheck', return_value=self.known))
        self.http = self.stack.enter_context(patch.object(engine, 'extract_with_requests_and_raw_html', return_value=(self.text, '<html/>')))
        self.pw = self.stack.enter_context(patch.object(engine, 'extract_with_playwright_and_raw_html', side_effect=RuntimeError('browser offline')))
        self.links = self.stack.enter_context(patch.object(engine, 'extract_deep_links', return_value=[]))
        self.warmup = self.stack.enter_context(patch.object(engine, 'warmup_engine'))

    def run_prediction(self):
        result = engine.predict_phishing_result(self.url)
        self.assertEqual(result['judgment'], 'unnormal')
        self.assertEqual(result['riskLevel'], 'HIGH')
        self.assertEqual(result['threat_score'], 100.0)
        self.assertEqual(set(result), set(self.known))
        self.assertEqual(set(result['evidence']), set(self.known['evidence']))
        for group in result['evidence']:
            self.assertEqual(set(result['evidence'][group]), set(self.known['evidence'][group]))
        self.assertEqual(result['evidence']['heuristic_evidence'], self.known['evidence']['heuristic_evidence'])
        self.assertEqual(self.known, self.original)  # Request-local report, no shared mutation.
        reason = result['evidence']['ai_semantic_evidence']['ai_inference_logic']
        self.assertTrue(reason.startswith('해당 사이트는 이미 기록되어있는 피싱 사이트입니다. 접속을 삼가해주세요.'))
        self.links.assert_not_called()  # No recursive traversal for an already known URL.
        return result, reason

    def test_listed_whitelisted_url_runs_model_once_and_keeps_raw_score(self):
        result, reason = self.run_prediction()
        self.http.assert_called_once_with(self.url)
        self.model.assert_called_once()
        self.warmup.assert_not_called()
        self.pw.assert_not_called()
        self.assertIn('88.1%', reason)
        self.assertIn('룰 보정 전', reason)
        self.assertIn('수집 본문 발췌', reason)
        self.assertEqual(self.tok.call_args.args[0], engine.prepare_text(self.text))
        self.assertFalse(self.model.call_args.kwargs['output_attentions'])

    def test_low_model_score_does_not_override_list_verdict(self):
        self.model.return_value.logits = torch.tensor([[20., -20.]])
        _, reason = self.run_prediction()
        self.assertIn('악성 클래스 점수는 0.0%', reason)

    def test_collection_uses_same_normalization_as_list_lookup(self):
        for url in ('naver.com/listed-path', 'hxxps://NAVER[.]com:443/listed-path'):
            with self.subTest(url=url):
                self.url = url
                self.http.reset_mock()
                self.run_prediction()
                self.http.assert_called_once_with('https://naver.com/listed-path')

    def test_error_page_and_form_rule_do_not_skip_model(self):
        for text in ('404 not found', 'Cloudflare Ray ID', 'Suspected phishing site', self.text):
            with self.subTest(text=text[:25]):
                self.http.return_value = (text, '<form action="https://api.telegram.org/example"></form>')
                self.model.reset_mock()
                self.run_prediction()
                self.model.assert_called_once()

    def test_unavailable_or_empty_text_does_not_fabricate_model_score(self):
        for text in ('', '   ', '[오류] Timeout', '[판별 보류] 수집 실패'):
            with self.subTest(text=text):
                self.http.return_value = (text, '')
                result, reason = self.run_prediction()
                self.assertIn('추가 분석 불가', reason)
                self.assertNotIn('악성 클래스 점수는', reason)
                self.model.assert_not_called()
                self.assertEqual(result['evidence']['ai_semantic_evidence']['suspect_sentence'], 'KoBERT 분석 문장 없음')

    def test_dynamic_fallback_recovers_network_failure(self):
        self.http.side_effect = OSError('HTTP offline')
        self.pw.side_effect = None
        self.pw.return_value = (self.text, '<html/>')
        with patch.dict(engine.os.environ, {'USE_PLAYWRIGHT_IN_ANALYZE': '1'}):
            _, reason = self.run_prediction()
        self.pw.assert_called_once_with(self.url, is_warmup=False)
        self.assertIn('88.1%', reason)

    def test_render_failure_preserves_usable_static_text(self):
        self.http.return_value = ('계좌번호를 제출해주세요.', '')
        with patch.dict(engine.os.environ, {'USE_PLAYWRIGHT_IN_ANALYZE': '1'}):
            _, reason = self.run_prediction()
        self.assertIn('88.1%', reason)
        self.pw.assert_called_once()

    def test_all_collection_failures_keep_list_verdict(self):
        self.http.side_effect = OSError('HTTP offline')
        with patch.dict(engine.os.environ, {'USE_PLAYWRIGHT_IN_ANALYZE': '1'}):
            _, reason = self.run_prediction()
        self.assertIn('추가 분석 불가', reason)
        self.model.assert_not_called()

    def test_inference_failure_keeps_list_verdict(self):
        self.model.side_effect = RuntimeError('inference failed')
        _, reason = self.run_prediction()
        self.assertIn('추론 실패', reason)
        self.assertNotIn('악성 클래스 점수는', reason)

    def test_invalid_logits_are_not_reported_as_probabilities(self):
        for logits in (torch.tensor([[float('nan'), 0.]]), torch.zeros((1, 3)),
                       torch.tensor([[0., float('inf')]])):
            with self.subTest(logits=logits):
                self.model.return_value.logits = logits
                _, reason = self.run_prediction()
                self.assertIn('추가 분석 불가', reason)
                self.assertNotIn('악성 클래스 점수는', reason)

    def test_cold_engine_warms_up_before_forward(self):
        with patch.object(engine, 'engine_initialized', False):
            self.run_prediction()
        self.warmup.assert_called_once_with(include_pw=False)
        self.model.assert_called_once()

    def test_unlisted_whitelisted_url_keeps_original_behavior(self):
        self.precheck.return_value = None
        result = engine.predict_phishing_result(self.url)
        self.assertEqual(result['judgment'], 'normal')
        self.http.assert_not_called()
        self.model.assert_not_called()


if __name__ == '__main__':
    unittest.main()
