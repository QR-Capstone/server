#!/usr/bin/env python3
"""Self-check phishing blocklist key generation, snapshot lookup, and engine lane."""

from __future__ import annotations

import gzip
import os
import sys
import tempfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL_ML_DIR = os.path.join(BASE, "url_ml")
for path in (BASE, URL_ML_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import phishing_blocklist  # noqa: E402
from phishing_blocklist import blocklist_key_hash, blocklist_keys_for_url  # noqa: E402


def check_key_generation() -> None:
    # Scheme, www, port, trailing slash, and case are normalized away.
    assert blocklist_keys_for_url("https://www.Evil-Site.com/Login/") == ["evil-site.com/login"]
    assert blocklist_keys_for_url("http://evil-site.com:80/login") == ["evil-site.com/login"]
    assert blocklist_keys_for_url("evil-site.com/login") == ["evil-site.com/login"]

    # Deep links with query also produce a query-stripped variant.
    keys = blocklist_keys_for_url("https://evil-site.com/kit?id=42")
    assert keys == ["evil-site.com/kit?id=42", "evil-site.com/kit"]

    # Root URLs with query must NOT produce a bare-host key (open redirects on
    # legitimate hosts would otherwise blocklist the whole host).
    keys = blocklist_keys_for_url("https://legit.com/?next=https://evil.com")
    assert keys == ["legit.com?next=https://evil.com"]

    # Userinfo lure does not leak the fake host into the key.
    assert blocklist_keys_for_url("https://bank.com@evil.com/x") == ["evil.com/x"]

    assert blocklist_keys_for_url("") == []
    assert blocklist_keys_for_url("   ") == []


def check_snapshot_lookup() -> None:
    entries = ["evil-site.com/login", "campaign-domain.top"]
    with tempfile.TemporaryDirectory() as tmp:
        snapshot = os.path.join(tmp, "snapshot.txt.gz")
        with gzip.open(snapshot, "wt", encoding="ascii") as f:
            f.write("# comment line\n")
            for key in entries:
                f.write(f"{blocklist_key_hash(key):016x}\n")

        old_path = phishing_blocklist.SNAPSHOT_PATH
        old_loaded = phishing_blocklist._loaded
        old_hashes = phishing_blocklist._hashes
        try:
            phishing_blocklist.SNAPSHOT_PATH = snapshot
            phishing_blocklist._loaded = False
            phishing_blocklist._hashes = frozenset()

            assert phishing_blocklist.blocklist_snapshot_size() == 2
            assert phishing_blocklist.is_blocklisted_url("https://www.evil-site.com/login/")
            assert phishing_blocklist.is_blocklisted_url("http://campaign-domain.top")
            # Query-stripped variant matches a listed deep link.
            assert phishing_blocklist.is_blocklisted_url("https://evil-site.com/login?session=99")
            # Different path or unlisted host must not match.
            assert not phishing_blocklist.is_blocklisted_url("https://evil-site.com/other")
            assert not phishing_blocklist.is_blocklisted_url("https://campaign-domain.top/deep")
            assert not phishing_blocklist.is_blocklisted_url("https://example.com")
        finally:
            phishing_blocklist.SNAPSHOT_PATH = old_path
            phishing_blocklist._loaded = old_loaded
            phishing_blocklist._hashes = old_hashes


def check_missing_snapshot_is_safe() -> None:
    old_path = phishing_blocklist.SNAPSHOT_PATH
    old_loaded = phishing_blocklist._loaded
    old_hashes = phishing_blocklist._hashes
    try:
        phishing_blocklist.SNAPSHOT_PATH = os.path.join(BASE, "blocklist", "does_not_exist.txt.gz")
        phishing_blocklist._loaded = False
        phishing_blocklist._hashes = frozenset()
        assert phishing_blocklist.blocklist_snapshot_size() == 0
        assert not phishing_blocklist.is_blocklisted_url("https://anything.com")
    finally:
        phishing_blocklist.SNAPSHOT_PATH = old_path
        phishing_blocklist._loaded = old_loaded
        phishing_blocklist._hashes = old_hashes


def check_engine_lane() -> None:
    from url_ml_engine import predict_url_ml, predict_url_ml_batch

    entries = ["lexically-benign-phish.com"]
    with tempfile.TemporaryDirectory() as tmp:
        snapshot = os.path.join(tmp, "snapshot.txt.gz")
        with gzip.open(snapshot, "wt", encoding="ascii") as f:
            for key in entries:
                f.write(f"{blocklist_key_hash(key):016x}\n")

        old_path = phishing_blocklist.SNAPSHOT_PATH
        old_loaded = phishing_blocklist._loaded
        old_hashes = phishing_blocklist._hashes
        try:
            phishing_blocklist.SNAPSHOT_PATH = snapshot
            phishing_blocklist._loaded = False
            phishing_blocklist._hashes = frozenset()

            single = predict_url_ml(None, "https://lexically-benign-phish.com")
            assert single["riskLevel"] == "DANGEROUS", single
            assert single["adjusted_by_rule"] is True
            assert "차단 목록" in single["adjustment_reason"]

            batch = predict_url_ml_batch(None, ["https://lexically-benign-phish.com"])
            assert batch[0]["riskLevel"] == "DANGEROUS", batch

            # Trusted official hosts bypass the blocklist lane entirely.
            trusted = predict_url_ml(None, "https://www.naver.com")
            assert trusted["riskLevel"] == "SAFE", trusted
        finally:
            phishing_blocklist.SNAPSHOT_PATH = old_path
            phishing_blocklist._loaded = old_loaded
            phishing_blocklist._hashes = old_hashes


def main_check() -> int:
    check_key_generation()
    check_snapshot_lookup()
    check_missing_snapshot_is_safe()
    check_engine_lane()
    print("PASS blocklist selfcheck")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_check())
