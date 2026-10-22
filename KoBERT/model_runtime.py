"""Load the trained KoBERT bundle locally, once during engine warmup."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch
from transformers import BertConfig, BertForSequenceClassification

if __package__:
    from .model_assets.preprocessing import PREPROCESS_VERSION, prepare_text
else:
    from model_assets.preprocessing import PREPROCESS_VERSION, prepare_text

ASSETS = Path(__file__).resolve().parent / 'model_assets'
LABELS = {'normal': 0, 'unnormal': 1}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_runtime(weights_file, device):
    """No Hugging Face download and no silent fallback to a different tokenizer."""
    weights_file = Path(weights_file)
    metadata = json.loads((ASSETS / 'training.json').read_text(encoding='utf-8'))
    if metadata.get('status') not in {'trained_not_tested', 'evaluated'}:
        raise ValueError('Incomplete KoBERT checkpoint')
    if metadata.get('labels') != LABELS or metadata.get('max_length') != 512:
        raise ValueError('Unsupported KoBERT labels or input length')
    if metadata.get('focus_enabled') is not False:
        raise ValueError('This engine expects a checkpoint trained without focus reordering')
    if (metadata.get('preprocessing') != PREPROCESS_VERSION or
            _sha256(ASSETS / 'preprocessing.py') != metadata.get('preprocessing_sha256')):
        raise ValueError('KoBERT preprocessing does not match the checkpoint')
    if _sha256(weights_file) != metadata.get('weights_sha256'):
        raise ValueError('KoBERT weights do not match training.json; download Git LFS files and deploy the matching pair')

    sources = json.loads((ASSETS / 'tokenizer' / 'sources.json').read_text(encoding='utf-8'))
    for name in ['tokenization_kobert.py', 'tokenizer_78b3253a26.model', 'vocab.txt']:
        if _sha256(ASSETS / 'tokenizer' / name) != sources[name]['sha256']:
            raise ValueError(f'KoBERT tokenizer hash mismatch: {name}')
    spec = importlib.util.spec_from_file_location(
        'server_kobert_tokenizer', ASSETS / 'tokenizer' / 'tokenization_kobert.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_path = ASSETS / 'tokenizer' / 'tokenizer_78b3253a26.model'
    temporary_assets = None
    if os.name == 'nt' and not str(model_path).isascii():
        temporary_assets = tempfile.TemporaryDirectory(prefix='kobert_sp_')
        model_path = Path(temporary_assets.name) / model_path.name
        shutil.copyfile(ASSETS / 'tokenizer' / model_path.name, model_path)
    tokenizer = module.KoBertTokenizer(
        vocab_file=str(model_path), vocab_txt=str(ASSETS / 'tokenizer' / 'vocab.txt'),
        do_lower_case=False, model_max_length=512)
    tokenizer._temporary_assets = temporary_assets
    if (tokenizer.vocab_size, tokenizer.unk_token_id, tokenizer.pad_token_id,
            tokenizer.cls_token_id, tokenizer.sep_token_id) != (8002, 0, 1, 2, 3):
        raise ValueError('Unexpected KoBERT vocabulary/token IDs')
    config = BertConfig.from_pretrained(ASSETS, local_files_only=True)
    if config.label2id != LABELS or config.vocab_size != tokenizer.vocab_size or config.pad_token_id != tokenizer.pad_token_id:
        raise ValueError('KoBERT config/tokenizer mismatch')
    # The engine uses attentions for evidence; retain the eager implementation.
    config._attn_implementation = 'eager'
    model = BertForSequenceClassification(config)
    state = torch.load(weights_file, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return tokenizer, model.to(device).eval()
