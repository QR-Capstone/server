"""Offline integration checks with the supplied checkpoint (requires Git LFS)."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
ENGINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE_DIR))
import torch
import koBERT as engine
import model_runtime


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with contextlib.redirect_stdout(io.StringIO()):
                engine.warmup_engine(False)
        # Authored test inputs; no training/validation dataset excerpts are published.
        cls.samples = [
            {'text': '박물관 전시 일정과 관람 시간을 안내합니다. 전시 해설은 안내 데스크에서 신청할 수 있습니다.'},
            {'text': '원금 보장 투자 안내입니다. 계좌번호와 인증번호를 제출하면 수익 보장 혜택을 드립니다.'},
        ]

    def test_training_validation_scores_reproduced(self):
        path = os.getenv('KOBERT_VALIDATION_SAMPLES')
        if not path:
            self.skipTest('Set KOBERT_VALIDATION_SAMPLES to a private validation fixture file')
        samples = json.loads(Path(path).read_text(encoding='utf-8'))
        texts = [model_runtime.prepare_text(row['text']) for row in samples]
        inputs = engine.tokenizer(texts, padding='max_length', max_length=512,
                                  truncation=True, return_tensors='pt')
        with torch.inference_mode():
            outputs = engine.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                   output_attentions=True)
        scores = torch.softmax(outputs.logits, dim=-1)[:, 1].tolist()
        self.assertIsNotNone(outputs.attentions)
        for sample, score in zip(samples, scores):
            self.assertAlmostEqual(score, sample['probability'], delta=1e-4)
            self.assertEqual(int(score > .5), sample['label'])

    def test_warmup_reuses_loaded_model(self):
        loaded = engine.model
        with patch.object(engine, 'load_runtime', side_effect=AssertionError('Unexpected reload')):
            self.assertEqual(engine.warmup_engine(False), {'status': 'already_initialized'})
        self.assertIs(engine.model, loaded)

    def test_html_preparation_once_and_old_masking_equivalence(self):
        html = '<html><title>고객 안내</title><body>' + ('전화 02-1234-5678 이메일 user@example.com 안내 ' * 80) + '</body></html>'
        with patch.object(engine, 'prepare_text', wraps=model_runtime.prepare_text) as prepare:
            result = engine.extract_with_html_ultimate_clean(html)
        prepare.assert_called_once()
        raw = prepare.call_args.args[0]
        old = engine.redact_pii(engine.re.sub(r'\s+', ' ', raw).strip()[:1000])
        self.assertEqual(result, old)

    def test_json_contract_and_no_keyword_prefix(self):
        for sample in self.samples:
            text = model_runtime.prepare_text(sample['text'])
            captured = []
            tokenizer = engine.tokenizer

            class RecordingTokenizer:
                def __call__(self, value, **kwargs):
                    captured.append(value)
                    return tokenizer(value, **kwargs)

                def tokenize(self, value):
                    return tokenizer.tokenize(value)

            with patch.dict(os.environ, {'USE_PLAYWRIGHT_IN_ANALYZE': '0', 'KOBERT_MAX_LEN': '512'}), \
                    patch.object(engine, 'extract_with_requests_and_raw_html', return_value=(text, '<html></html>')), \
                    patch.object(engine, 'extract_with_playwright_and_raw_html', side_effect=AssertionError('Network forbidden')), \
                    patch.object(engine, 'extract_deep_links', return_value=[]), \
                    patch.object(engine, 'tokenizer', RecordingTokenizer()), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = engine.predict_phishing_result('https://runtime-check.test/')
            self.assertEqual(captured, [text])
            self.assertEqual(set(result), {'url', 'judgment', 'riskLevel', 'threat_score',
                                          'threat_type', 'site_category', 'evidence'})
            self.assertIn(result['judgment'], {'normal', 'unnormal'})
            self.assertEqual(result['riskLevel'], 'HIGH' if result['judgment'] == 'unnormal' else 'LOW')
            self.assertIsInstance(result['threat_score'], float)
            self.assertEqual(set(result['evidence']), {'heuristic_evidence', 'ai_semantic_evidence'})
            self.assertEqual(set(result['evidence']['ai_semantic_evidence']), {'suspect_sentence', 'ai_inference_logic'})
            self.assertEqual(set(result['evidence']['heuristic_evidence']), {'detected_actions', 'rule_trigger'})

    def test_failed_collection_keeps_unknown_contract(self):
        with patch.dict(os.environ, {'USE_PLAYWRIGHT_IN_ANALYZE': '0'}), \
                patch.object(engine, 'extract_with_requests_and_raw_html', return_value=('[오류] 연결 실패', '')), \
                contextlib.redirect_stdout(io.StringIO()):
            result = engine.predict_phishing_result('https://runtime-check.test/')
        self.assertEqual(result['judgment'], 'unknown')
        self.assertEqual(result['riskLevel'], 'UNKNOWN')

    def test_wrong_weights_rejected_before_deserialization(self):
        real_sha = model_runtime._sha256
        def sha(path):
            return 'wrong' if str(path).endswith('.pt') else real_sha(path)
        with patch.object(model_runtime, '_sha256', side_effect=sha), \
                patch.object(torch, 'load', side_effect=AssertionError('Must reject before loading')):
            with self.assertRaisesRegex(ValueError, 'weights do not match'):
                model_runtime.load_runtime(engine.WEIGHTS_FILE, 'cpu')


if __name__ == '__main__':
    unittest.main()
