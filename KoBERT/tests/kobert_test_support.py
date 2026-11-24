"""Offline engine harness for regression tests; no local desktop runner dependency."""
import contextlib
import io
import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import koBERT


class LocalTester:
    def __init__(self, enable_feed=False):
        self.engine = koBERT
        torch.set_num_threads(4)
        with patch.dict(os.environ, {'KOBERT_NURILAB_ENABLED': '0', 'HF_HUB_OFFLINE': '1',
                                     'TRANSFORMERS_OFFLINE': '1'}), contextlib.redirect_stdout(io.StringIO()):
            koBERT.warmup_engine(include_pw=False)

    def url(self, value, browser=True):
        with patch.dict(os.environ, {'USE_PLAYWRIGHT_IN_ANALYZE': '1' if browser else '0',
                                     'KOBERT_MAX_LEN': '512'}), contextlib.redirect_stdout(io.StringIO()):
            return koBERT.predict_phishing_result(value)

    def text(self, value):
        prepared = koBERT.prepare_focus_text(value)
        inputs = koBERT.tokenizer(prepared, max_length=512, padding='max_length',
                                  truncation=True, return_tensors='pt')
        with torch.inference_mode():
            result = koBERT.model(input_ids=inputs['input_ids'].to(koBERT.device),
                                  attention_mask=inputs['attention_mask'].to(koBERT.device))
            score = torch.softmax(result.logits, dim=-1)[0, 1].item()
        return {'mode': 'text_model_only', 'judgment': 'unnormal' if score > .5 else 'normal',
                'phishing_score': round(score, 6), 'threshold': .5,
                'input_characters': len(value), 'prepared_characters': len(prepared)}
