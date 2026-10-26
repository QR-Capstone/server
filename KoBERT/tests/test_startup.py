"""Initialization failure and concurrency checks without fetching any URLs."""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import koBERT as engine


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.state = patch.multiple(engine, engine_initialized=False,
                                    device=None, tokenizer=None, model=None,
                                    async_pw_manager=None)
        self.state.start()
        self.addCleanup(self.state.stop)
        self.cuda = patch.object(torch.cuda, 'is_available', return_value=False)
        self.cuda.start()
        self.addCleanup(self.cuda.stop)
        self.tokenizer = Mock(return_value={
            'input_ids': torch.zeros((1, 512), dtype=torch.long),
            'attention_mask': torch.ones((1, 512), dtype=torch.long)})
        self.model = Mock(return_value=SimpleNamespace(logits=torch.zeros((1, 2))))

    def test_concurrent_startup_loads_only_once(self):
        with patch.object(engine, 'load_runtime', return_value=(self.tokenizer, self.model)) as load, \
                contextlib.redirect_stdout(io.StringIO()):
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda _: engine.warmup_engine(False), range(8)))
        load.assert_called_once()
        self.model.assert_called_once()
        self.assertEqual(sum(result.get('model_loaded', False) for result in results), 1)
        self.assertTrue(engine.engine_initialized)
        self.assertIs(engine.model, self.model)
        self.assertEqual(self.tokenizer.call_args.kwargs['max_length'], 512)
        self.assertTrue(self.model.call_args.kwargs['output_attentions'])

    def test_failed_forward_leaves_uninitialized_and_can_retry(self):
        self.model.side_effect = RuntimeError('forward failed')
        with patch.object(engine, 'load_runtime', return_value=(self.tokenizer, self.model)), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'forward failed'):
                engine.warmup_engine(False)
            self.assertFalse(engine.engine_initialized)
            self.assertIsNone(engine.model)
            self.assertIsNone(engine.tokenizer)
            self.model.side_effect = None
            engine.warmup_engine(False)
        self.assertTrue(engine.engine_initialized)

    def test_bad_assets_leave_uninitialized(self):
        with patch.object(engine, 'load_runtime', side_effect=ValueError('asset mismatch')), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'asset mismatch'):
                engine.warmup_engine(False)
        self.assertFalse(engine.engine_initialized)
        self.assertIsNone(engine.model)

    def test_nonfinite_or_wrong_shape_logits_are_not_ready(self):
        for logits in (torch.tensor([[float('nan'), 0.0]]), torch.zeros((1, 3))):
            with self.subTest(shape=tuple(logits.shape)):
                self.model.return_value = SimpleNamespace(logits=logits)
                with patch.object(engine, 'load_runtime', return_value=(self.tokenizer, self.model)), \
                        contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(RuntimeError):
                        engine.warmup_engine(False)
                self.assertFalse(engine.engine_initialized)
                self.assertIsNone(engine.model)


if __name__ == '__main__':
    unittest.main()
