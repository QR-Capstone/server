#!/usr/bin/env python3
"""Self-check batch URL dedupe and response-order metadata."""

from __future__ import annotations

import os
import sys

from fastapi import HTTPException


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import main  # noqa: E402


def main_check() -> int:
    urls = [
        "Example.com/login",
        " https://example.com/login ",
        "https://example.com/login?x=1",
        "https://EXAMPLE.com/login?x=1",
        "http://example.com/login",
        "https://example.com/login#frag",
        "https://example.com/login#frag",
    ]
    normalized, keys, unique_urls, first_index = main._prepare_batch_urls(urls)

    assert normalized == [url.strip() for url in urls]
    assert keys[0] == "https://example.com/login"
    assert keys[1] == keys[0]
    assert keys[2] == "https://example.com/login?x=1"
    assert keys[3] == keys[2]
    assert keys[4] == "http://example.com/login"
    assert keys[5] == "https://example.com/login#frag"
    assert keys[6] == keys[5]

    assert list(unique_urls) == [keys[0], keys[2], keys[4], keys[5]]
    assert unique_urls[keys[0]] == "Example.com/login"
    assert unique_urls[keys[2]] == "https://example.com/login?x=1"
    assert first_index[keys[0]] == 0
    assert first_index[keys[2]] == 2
    assert first_index[keys[5]] == 5

    duplicate_of = [
        first_index[key] if first_index[key] != index else None
        for index, key in enumerate(keys)
    ]
    assert duplicate_of == [None, 0, None, 2, None, None, 5], duplicate_of

    try:
        main._prepare_batch_urls(["https://example.com", " "])
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "index 1" in str(exc.detail)
    else:
        raise AssertionError("empty batch URL did not raise HTTPException")

    print("batch dedupe selfcheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_check())
