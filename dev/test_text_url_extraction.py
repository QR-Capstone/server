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
    reversed_malicious = "nigol/nc.moc.spw-eciiffo-sppa.www//:sptth"
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
    assert_extracts(malicious.encode("utf-8").hex(), malicious)
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
    assert_extracts("https%253A%252F%252Fwww.apps-offiice-wps.com.cn%252Flogin", malicious)
    assert_extracts(
        r"\u{68}\u{74}\u{74}\u{70}\u{73}\u{3a}\u{2f}\u{2f}www.apps-offiice-wps.com.cn/login",
        malicious,
    )
    assert_extracts(
        r"\68\74\74\70\73\3a\2f\2fwww\2eapps-offiice-wps\2ecom\2ecn\2flogin",
        malicious,
    )
    assert_extracts(
        r"\000068\000074\000074\000070\000073\00003a\00002f\00002fwww"
        r"\00002eapps-offiice-wps\00002ecom\00002ecn\00002flogin",
        malicious,
    )
    assert_extracts(
        r"\150\164\164\160\163\072\057\057www\056apps-offiice-wps\056com\056cn\057login",
        malicious,
    )
    assert_extracts(
        r"\150\164\164\160\163://www\056apps-offiice-wps\056com\056cn/login",
        malicious,
    )
    assert_extracts(
        "location=String.fromCharCode("
        "104,116,116,112,115,58,47,47,119,119,119,46,97,112,112,115,45,111,102,102,105,105,99,101,"
        "45,119,112,115,46,99,111,109,46,99,110,47,108,111,103,105,110)",
        malicious,
    )
    assert_extracts(
        "location=String.fromCharCode("
        "0x68,0x74,0x74,0x70,0x73,0x3a,0x2f,0x2f,0x77,0x77,0x77,0x2e,0x61,0x70,0x70,0x73,"
        "0x2d,0x6f,0x66,0x66,0x69,0x69,0x63,0x65,0x2d,0x77,0x70,0x73,0x2e,0x63,0x6f,0x6d,"
        "0x2e,0x63,0x6e,0x2f,0x6c,0x6f,0x67,0x69,0x6e)",
        malicious,
    )
    assert_extracts(
        """location=['hxxps[:]//www','[.]apps-offiice-wps','[.]com','[.]cn/login'].join('')""",
        malicious,
    )
    assert_extracts(
        "location='nigol/nc.moc.spw-eciiffo-sppa.www//:sptth'.split('').reverse().join('')",
        malicious,
    )
    assert_not_extracts(
        "location='nigol/nc.moc.spw-eciiffo-sppa.www//:sptth'.split('').reverse().join('')",
        reversed_malicious,
    )
    assert_extracts(
        """location=['nigol/nc.m','oc.spw-eciiffo-','sppa.www//:sptth'].join('').split('').reverse().join('')""",
        malicious,
    )
    assert_extracts(
        "go hxxps://login-example[.]shop[@]evil-checkout[.]test/path",
        "https://login-example.shop@evil-checkout.test/path",
    )
    assert_extracts(
        "go https://redirector.test/click?target=https%253A%252F%252Fwww.apps-offiice-wps.com.cn%252Flogin",
        malicious,
    )
    assert_extracts(
        "go https://redirector.test/#/login?redirect=https%253A%252F%252Fwww.apps-offiice-wps.com.cn%252Flogin",
        malicious,
    )
    assert_extracts(
        "go https://redirector.test/click?url=aHR0cHM6Ly93d3cuYXBwcy1vZmZpaWNlLXdwcy5jb20uY24vbG9naW4",
        malicious,
    )
    assert_extracts(
        "go https://redirector.test/click?url=aHR0cHM6Ly93d3cuYXBwcy1vZmZpaWNlLXdwcy5jb20uY24vbG9naW4%253D",
        malicious,
    )
    assert_extracts(
        "see (https://www.apps-offiice-wps.com.cn/path_(x))",
        "https://www.apps-offiice-wps.com.cn/path_(x)",
    )
    assert_not_extracts(
        "read https://www.apps-offiice-wps.com.cn/ now",
        "https://www.apps-offiice-wps.com.cn/now",
    )
    print("PASS text url extraction self-check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_check())
