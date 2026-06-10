#!/usr/bin/env python3
"""Self-checks for KoBERT URL fallback text generation."""

from __future__ import annotations

from collect_kobert_text_dataset import fallback_text


def main() -> int:
    url = "http://www.11st-cert.kr-545285.xyz/index.php?mode=cert"
    malicious_source_text = fallback_text(url, "malicious_real_live")
    benign_source_text = fallback_text(url, "benign_major_official")
    assert malicious_source_text == benign_source_text
    assert "데이터 출처" not in malicious_source_text
    assert "malicious" not in malicious_source_text.lower()
    assert "benign" not in malicious_source_text.lower()
    assert "URL 구조 신호" in malicious_source_text
    assert "암호화되지 않은 HTTP" in malicious_source_text
    assert "긴 숫자 포함 도메인" in malicious_source_text

    trusted_text = fallback_text("https://trustwallet.com", "malicious_real_live")
    assert "데이터 출처" not in trusted_text
    assert "공식 신뢰 도메인" in trusted_text
    assert "지갑 브랜드명 포함" not in trusted_text

    print("PASS kobert text dataset selfcheck")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
