#!/usr/bin/env python3
"""Self-check API URL rule adjustment for embedded redirect targets."""

from __future__ import annotations

import os
import sys


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import main  # noqa: E402


def main_check() -> int:
    embedded_malicious = (
        "https://redirector.test/click?"
        "target=https%253A%252F%252Fwww.apps-offiice-wps.com.cn%252Flogin"
    )
    adjustment = main._url_rule_adjustment(main._url_cache_key(embedded_malicious))
    assert adjustment, "redirect target adjustment missing"
    assert adjustment["riskLevel"] == "DANGEROUS", adjustment
    assert "redirect" in adjustment["reason"], adjustment

    fragment_redirect = (
        "https://redirector.test/#/login?"
        "redirect=https%253A%252F%252Fwww.apps-offiice-wps.com.cn%252Flogin"
    )
    fragment_adjustment = main._url_rule_adjustment(main._url_cache_key(fragment_redirect))
    assert fragment_adjustment, "fragment redirect target adjustment missing"
    assert fragment_adjustment["riskLevel"] == "DANGEROUS", fragment_adjustment

    base64_redirect = (
        "https://redirector.test/click?"
        "url=aHR0cHM6Ly93d3cuYXBwcy1vZmZpaWNlLXdwcy5jb20uY24vbG9naW4"
    )
    base64_adjustment = main._url_rule_adjustment(main._url_cache_key(base64_redirect))
    assert base64_adjustment, "base64 redirect target adjustment missing"
    assert base64_adjustment["riskLevel"] == "DANGEROUS", base64_adjustment

    double_encoded_base64_redirect = (
        "https://redirector.test/click?"
        "url=aHR0cHM6Ly93d3cuYXBwcy1vZmZpaWNlLXdwcy5jb20uY24vbG9naW4%253D"
    )
    double_encoded_base64_adjustment = main._url_rule_adjustment(
        main._url_cache_key(double_encoded_base64_redirect)
    )
    assert double_encoded_base64_adjustment, "double-encoded base64 redirect target adjustment missing"
    assert double_encoded_base64_adjustment["riskLevel"] == "DANGEROUS", double_encoded_base64_adjustment

    hex_redirect = (
        "https://redirector.test/click?url="
        + "https://www.apps-offiice-wps.com.cn/login".encode("utf-8").hex()
    )
    hex_adjustment = main._url_rule_adjustment(main._url_cache_key(hex_redirect))
    assert hex_adjustment, "hex redirect target adjustment missing"
    assert hex_adjustment["riskLevel"] == "DANGEROUS", hex_adjustment

    css_hex_redirect = (
        "https://redirector.test/click?"
        r"url=%5C68%5C74%5C74%5C70%5C73%5C3a%5C2f%5C2fwww%5C2eapps-offiice-wps%5C2ecom%5C2ecn%5C2flogin"
    )
    css_hex_adjustment = main._url_rule_adjustment(main._url_cache_key(css_hex_redirect))
    assert css_hex_adjustment, "CSS hex escaped redirect target adjustment missing"
    assert css_hex_adjustment["riskLevel"] == "DANGEROUS", css_hex_adjustment

    css_zero_padded_redirect = (
        "https://redirector.test/click?"
        r"url=%5C000068%5C000074%5C000074%5C000070%5C000073%5C00003a%5C00002f%5C00002fwww"
        r"%5C00002eapps-offiice-wps%5C00002ecom%5C00002ecn%5C00002flogin"
    )
    css_zero_padded_adjustment = main._url_rule_adjustment(main._url_cache_key(css_zero_padded_redirect))
    assert css_zero_padded_adjustment, "zero-padded CSS hex escaped redirect target adjustment missing"
    assert css_zero_padded_adjustment["riskLevel"] == "DANGEROUS", css_zero_padded_adjustment

    js_octal_redirect = (
        "https://redirector.test/click?"
        r"url=%5C150%5C164%5C164%5C160%5C163%5C072%5C057%5C057www"
        r"%5C056apps-offiice-wps%5C056com%5C056cn%5C057login"
    )
    js_octal_adjustment = main._url_rule_adjustment(main._url_cache_key(js_octal_redirect))
    assert js_octal_adjustment, "JS octal escaped redirect target adjustment missing"
    assert js_octal_adjustment["riskLevel"] == "DANGEROUS", js_octal_adjustment

    userinfo = "https://trustwallet.com@evil-checkout.test/connect"
    userinfo_adjustment = main._url_rule_adjustment(main._url_cache_key(userinfo))
    assert userinfo_adjustment, "userinfo lure adjustment missing"
    assert userinfo_adjustment["riskLevel"] == "DANGEROUS", userinfo_adjustment

    trusted = "https://trustwallet.com/?target=https%3A%2F%2Ftrustwallet.com%2F"
    assert main._url_rule_adjustment(main._url_cache_key(trusted))["riskLevel"] == "SAFE"

    print("url rule adjustment selfcheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_check())
