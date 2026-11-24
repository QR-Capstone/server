"""Offline tests for the separate keyword-selection experiment."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import koBERT as engine
from focus_selection import KEYWORDS, has_focus_keyword, select_keyword_context


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'KOBERT_FOCUS_ENABLED': '1'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_no_keyword_is_exact_original_prefix(self):
        parts = [f'전시실 {i}의 안내 내용입니다.' for i in range(100)]
        self.assertEqual(select_keyword_context(parts), ' '.join(parts)[:500])

    def test_keyword_after_old_cut_is_included(self):
        parts = [f'전시실 {i}의 안내 내용입니다.' for i in range(100)]
        parts += ['계정 정지를 해제하려면 정보를 입력하세요.', '요청 전에 운영자를 확인하세요.']
        self.assertNotIn('계정 정지', ' '.join(parts)[:500])
        selected = select_keyword_context(parts)
        self.assertIn('계정 정지', selected)
        self.assertIn('운영자를 확인하세요', selected)
        self.assertLessEqual(len(selected), 500)

    def test_previous_and_next_sentences_and_negation_kept(self):
        selected = select_keyword_context([
            '다음은 사기 예방 안내입니다.', '원금 보장을 내세운 입금 요구에 응하지 마세요.',
            '공식 고객센터를 통해 확인하세요.', '일반적인 전시 소식입니다.'])
        self.assertIn('사기 예방', selected)
        self.assertIn('응하지 마세요', selected)
        self.assertIn('공식 고객센터', selected)
        self.assertNotIn('전시 소식', selected)

    def test_long_sentence_window_contains_late_keyword(self):
        text = '일반 안내 ' * 200 + '수익 보장을 내세운 요구에 응하지 마세요.'
        selected = select_keyword_context([text])
        self.assertIn('수익 보장', selected)
        self.assertIn('응하지 마세요', selected)
        self.assertLessEqual(len(selected), 500)

    def test_case_spacing_and_repeat_windows(self):
        self.assertTrue(has_focus_keyword('vvip   정보 안내'))
        text = '원금 보장과 수익 보장 안내입니다. 응하지 마세요.'
        selected = select_keyword_context([text])
        self.assertEqual(selected.count('원금 보장'), 1)

    def test_overlapping_neighbours_do_not_discard_next_keyword_sentence(self):
        selected = select_keyword_context(['원금 보장 안내입니다.', '중간 안내입니다.',
                                           '계정 정지 안내입니다.', '마지막 안내입니다.'])
        self.assertIn('원금 보장', selected)
        self.assertIn('계정 정지', selected)
        self.assertEqual(selected.count('중간 안내입니다.'), 1)

    def test_disabled_mode_matches_original_character_cap(self):
        text = '메뉴 안내 ' * 200 + '원금 보장 안내.'
        with patch.dict(os.environ, {'KOBERT_FOCUS_ENABLED': '0'}):
            self.assertEqual(select_keyword_context([text]), text[:500])
            self.assertEqual(engine.prepare_focus_text(text), engine.prepare_text(text))

    def test_pii_masked_before_keyword_window_is_cut(self):
        text = '일반 안내 ' * 120 + '문의 010-1234-5678 원금 보장 안내입니다.'
        selected = select_keyword_context([text], masker=engine.redact_pii)
        self.assertNotIn('010-1234-5678', selected)
        self.assertIn('[고객센터_연락처]', selected)

    def test_title_and_form_information_still_in_html_input(self):
        html = '<title>고객 안내</title><body><p>원금 보장 안내입니다.</p><input type="password"><button>제출</button></body>'
        prepared = engine.extract_with_html_ultimate_clean(html)
        for text in ('고객 안내', '원금 보장', '비밀번호', '제출'):
            self.assertIn(text, prepared)

    def test_long_risk_only_html_candidate_is_not_discarded(self):
        text = '일반 안내 ' * 160 + '원금 보장 안내입니다.'
        html = '<body><p>' + text + '</p></body>'
        with patch.dict(os.environ, {'KOBERT_FOCUS_ENABLED': '0'}):
            baseline = engine.extract_with_html_ultimate_clean(html)
        self.assertNotIn('원금 보장', baseline)
        self.assertIn('원금 보장', engine.extract_with_html_ultimate_clean(html))

    def test_cache_keys_separate_modes(self):
        with patch.dict(os.environ, {'KOBERT_FOCUS_ENABLED': '0'}):
            baseline = engine._result_cache_key('https://example.test/')
        self.assertNotEqual(baseline, engine._result_cache_key('https://example.test/'))



class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kobert_test_support import LocalTester
        env = patch.dict(os.environ, {'KOBERT_RESULT_CACHE_ENABLED': '0', 'KOBERT_FOCUS_ENABLED': '1'})
        env.start()
        cls.addClassCleanup(env.stop)
        with contextlib.redirect_stderr(io.StringIO()):
            cls.tester = LocalTester(enable_feed=False)

    def test_focus_input_reaches_real_model_once_with_same_json_schema(self):
        text = '일반적인 전시 안내입니다. ' * 100 + '계정 정지 해제를 위해 인증번호를 입력하세요.'
        expected = engine.prepare_focus_text(text)
        real_tok = engine.tokenizer
        captured = []

        def recording_tokenizer(value, **kwargs):
            captured.append(value)
            return real_tok(value, **kwargs)

        with patch.object(engine, 'tokenizer', side_effect=recording_tokenizer), \
                patch.object(engine.model, 'forward', wraps=engine.model.forward) as forward, \
                contextlib.redirect_stderr(io.StringIO()):
            result = self.tester.text(text)
        forward.assert_called_once()
        self.assertEqual(captured, [expected])
        self.assertIn('계정 정지', captured[0])
        self.assertEqual(set(result), {'mode', 'judgment', 'phishing_score', 'threshold',
                                      'input_characters', 'prepared_characters'})


if __name__ == '__main__':
    unittest.main()
