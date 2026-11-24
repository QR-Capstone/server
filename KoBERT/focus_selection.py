"""Select keyword context before truncation; checkpoint and rule weights stay unchanged."""
import bisect
import json
import os
from pathlib import Path
import re

# A separate editable copy of the existing engine's high_risk_keywords.
KEYWORDS = json.loads(Path(__file__).with_name('focus_keywords.json').read_text(encoding='utf-8-sig'))
if not isinstance(KEYWORDS, list) or not KEYWORDS or not all(isinstance(k, str) and k.strip() for k in KEYWORDS):
    raise ValueError('focus_keywords.json must contain a non-empty list of keyword strings')

_PATTERN = re.compile('|'.join(re.escape(k.strip()).replace(r'\ ', r'\s*')
                             for k in sorted(set(KEYWORDS), key=len, reverse=True)), re.I)
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?。！？])\s+|\n+')


def focus_enabled():
    return os.getenv('KOBERT_FOCUS_ENABLED', '1') == '1'


def has_focus_keyword(text):
    return focus_enabled() and _PATTERN.search(text) is not None


def select_keyword_context(parts, limit=500, masker=lambda text: text):
    """Keep up to three keyword windows, with previous/next sentence when space permits.

    Windows retain document order. Oversized windows are cropped around the hit,
    before the old leading-character cut can discard it. No-hit/off is byte-for-byte
    equivalent to joining the original candidates and applying the old limit.
    """
    original = ' '.join(parts)
    fallback = original[:limit]
    if limit <= 0:
        return ''
    if not focus_enabled() or not _PATTERN.search(original):
        return fallback
    # Mask entire selected-source text before clipping, so contact numbers are not
    # cut into unrecognizable fragments before the existing PII masker sees them.
    text = masker(original)
    matches = list(_PATTERN.finditer(text))
    if not matches:
        return fallback
    boundaries = list(_SENTENCE_BOUNDARY.finditer(text))
    starts = [0] + [boundary.end() for boundary in boundaries]
    ends = [boundary.start() for boundary in boundaries] + [len(text)]
    windows = []
    for match in matches:
        index = min(bisect.bisect_right(starts, match.start()) - 1, len(starts) - 1)
        start = starts[max(0, index - 1)]
        end = ends[min(len(ends) - 1, index + 1)]
        if windows and match.start() < windows[-1][1]:
            # Repeated keywords in the same neighbourhood must not duplicate input
            # or grow the first window to consume the whole page.
            continue
        if windows:
            start = max(start, windows[-1][1])
        windows.append((start, end, match.start(), match.end()))
        if len(windows) == 3:
            break
    available = max(0, limit - (len(windows) - 1))
    base, remainder = divmod(available, len(windows))
    selected = []
    for i, (start, end, hit_start, hit_end) in enumerate(windows):
        budget = base + (1 if i < remainder else 0)
        if end - start > budget:
            left = max(start, hit_start - budget // 3)
            left = min(left, max(start, end - budget))
            if hit_end > left + budget:
                left = max(start, hit_end - budget)
            start, end = left, min(end, left + budget)
        selected.append(text[start:end].strip())
    return ' '.join(selected)[:limit]
