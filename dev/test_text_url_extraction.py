#!/usr/bin/env python3
"""Self-checks for URL extraction from obfuscated message text."""

from __future__ import annotations

import base64
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import main  # noqa: E402


def assert_extracts(text: str, expected: str) -> None:
    urls = main._extract_urls_from_text(text)
    if expected not in urls:
        raise AssertionError(f"expected {expected!r} in {urls!r} from {text!r}")


def assert_not_extracts(text: str, unexpected: str) -> None:
    urls = main._extract_urls_from_text(text)
    if unexpected in urls:
        raise AssertionError(f"unexpected {unexpected!r} in {urls!r} from {text!r}")


def main_check() -> int:
    malicious = "https://www.apps-offiice-wps.com.cn/login"
    assert_extracts("go https[:]//www[.]apps-offiice-wps[.]com[.]cn/login", malicious)
    assert_extracts("go hxxps(:)//www(.)apps-offiice-wps(.)com(.)cn/login", malicious)
    assert_extracts(
        "go hxxps \uc30d\uc810 \uc2ac\ub798\uc2dc \uc2ac\ub798\uc2dc "
        "www \uc810 apps-offiice-wps \uc810 com \uc810 cn/login",
        malicious,
    )
    assert_extracts('"https://www.apps" + "-offiice-wps.com.cn/login"', malicious)
    assert_extracts(
        base64.b64encode(b"hxxps[:]//www[.]apps-offiice-wps[.]com[.]cn/login").decode("ascii"),
        malicious,
    )
    assert_extracts(
        "\uff48\uff54\uff54\uff50\uff53\uff1a\uff0f\uff0f\uff57\uff57\uff57"
        "\uff0eapps-offiice-wps\uff0ecom\uff0ecn\uff0flogin",
        malicious,
    )
    assert_extracts(
        "hxxps \ucf5c\ub860 \uc2ac\ub798\uc2dc \uc2ac\ub798\uc2dc "
        "www \ub2f7 apps-offiice-wps \ub2f7 com \ub2f7 cn/login",
        malicious,
    )
    assert_extracts('"https://www.apps-offiice-" \\\n "wps.com.cn/login"', malicious)
    assert_extracts("https=3A=2F=2Fwww.apps-offiice-wps.com.cn=2Flogin", malicious)
    assert_not_extracts(
        "read https://www.apps-offiice-wps.com.cn/ now",
        "https://www.apps-offiice-wps.com.cn/now",
    )
    print("PASS text url extraction self-check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_check())
