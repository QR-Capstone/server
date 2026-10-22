"""Shared text preprocessing for the new checkpoint only (no network or model)."""
import re

PREPROCESS_VERSION = 'redact-whitespace-cap1000-v1-no-focus'
_PHONE = re.compile(r'\b\d{2,3}[-\s]?\d{3,4}[-\s]?\d{4}\b')
_EMAIL = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')

def prepare_text(text):
    # Input is already extracted page text, not raw HTML. This mirrors the
    # final whitespace/character cap/redaction stages of the current collector.
    text = re.sub(r'\s+', ' ', str(text)).strip()[:1000]
    text = _PHONE.sub('[고객센터_연락처]', text)
    text = _EMAIL.sub('[고객센터_Email]', text)
    text = re.sub(r'\b[\d\s]*\*{2,}[\d\s]*\b', '[카드번호_형태]', text)
    text = re.sub(r'\b\d{9,}\b', '[장문숫자]', text)
    return re.sub(r'http[s]?://(?:bit\.ly|vo\.la|t\.ly|cutt\.ly|url\.kr|ko\.gl|han\.gl|buly\.kr)/[a-zA-Z0-9]+', '[링크]', text)
