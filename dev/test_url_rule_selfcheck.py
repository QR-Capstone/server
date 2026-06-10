#!/usr/bin/env python3
"""Self-checks for conservative URL-only phishing rules."""

from __future__ import annotations

import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from trusted_domains import strong_url_phishing_score, url_heuristic_phishing_score


def main() -> int:
    malicious_roots = [
        "http://gobox.kr/",
        "http://trustwallet-web.at",
        "https://amazon1122.com",
        "https://freedomaston.com",
        "https://jx-instagram.com",
        "http://bradescosaudeconvenios.com.br",
        "http://url414.ucf.edu/ls/click?upn=u001.LlD2eeb7FrRssjXcZFaIAMnowgiUb-2B1Tx1A2RKmSgQNLJBfP6NaPquBjNQwWZcwctFZN_Z1dd0htOOzW9IbQkDigg21O85PkWPViQFI1pxYS26jhRVslTzucNk-2Brr0OlXHVFv6EQiIlwVMKwZP3N2OfqlxSnAj8pg2-2FqQIwVIhWNb7nAJbh4liBWTOAWkSm6PTRK33hSyHB4LJo7frdZ-2FvavfiQKu",
        "https://nkaeklkub.us16.list-manage.com/track/click?u=e383ad4a4a5b7ac07eb70f0be&id=041a26f896&e=bb0fa0d0f0",
        "https://trustwallet.com@evil-checkout.test/connect",
        "https://login.microsoftonline.com@evil-checkout.test/login",
        "https://www.pekj1403.com/aianl",
        "http://134744072/",
        "http://0x08080808/",
        "http://010.010.010.010/",
        "https://xn--80ak6aa92e.com/",
        "https://xn--l-7sba6dbr.com/",
        "https://confirm-payment-id694787.com",
        "https://www.fms-77.com/",
    ]
    for url in malicious_roots:
        strong = strong_url_phishing_score(url)
        heuristic = url_heuristic_phishing_score(url)
        assert strong >= 0.66, f"strong score too low for {url}: {strong}"
        assert heuristic >= 0.66, f"heuristic score too low for {url}: {heuristic}"

    trusted_roots = [
        "https://trustwallet.com",
        "https://www.trustwallet.com/",
        "https://eventmaster.ie",
        "https://rakuten-edy.co.jp",
    ]
    for url in trusted_roots:
        strong = strong_url_phishing_score(url)
        heuristic = url_heuristic_phishing_score(url)
        assert strong == 0.0, f"trusted strong score changed for {url}: {strong}"
        assert heuristic == 0.0, f"trusted heuristic score changed for {url}: {heuristic}"

    print("PASS url rule selfcheck")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
