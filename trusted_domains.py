"""Shared trusted-domain helpers for Korean phishing detection models."""

from __future__ import annotations

from urllib.parse import urlsplit
import re


TRUSTED_SUFFIXES = (
    ".go.kr",
    ".ac.kr",
    ".edu",
    ".mil.kr",
    ".ms.kr",
    ".or.kr",
)


TRUSTED_REGISTERED_DOMAINS = frozenset(
    {
        "naver.com",
        "navercorp.com",
        "pstatic.net",
        "kakao.com",
        "kakaocorp.com",
        "daum.net",
        "tistory.com",
        "kakaopay.com",
        "payco.com",
        "toss.im",
        "karrotpay.com",
        "kbstar.com",
        "shinhan.com",
        "wooribank.com",
        "hanabank.com",
        "kebhana.com",
        "kakaobank.com",
        "tossbank.com",
        "kbanknow.com",
        "ibk.co.kr",
        "nhbank.co.kr",
        "hyundaicard.com",
        "lottecard.co.kr",
        "samsungcard.com",
        "kbcard.com",
        "shinhancard.com",
        "nts.go.kr",
        "nhis.or.kr",
        "dhlottery.co.kr",
        "betman.co.kr",
        "coupang.com",
        "11st.co.kr",
        "hyundai.com",
        "lotte.com",
        "shinsegae.com",
        "auction.co.kr",
        "gmarket.co.kr",
        "interpark.com",
        "tmon.co.kr",
        "wemakeprice.com",
        "emart.com",
        "emart24.co.kr",
        "homeplus.co.kr",
        "lottemart.com",
        "lotteon.com",
        "lottcinema.co.kr",
        "lottechilsung.co.kr",
        "hyundaidepartment.com",
        "lottedepartment.com",
        "gsshop.com",
        "cjonstyle.com",
        "bgfretail.com",
        "gs25.com",
        "cu.co.kr",
        "ministop.co.kr",
        "cj.net",
        "cj.co.kr",
        "hanjin.com",
        "korail.com",
        "airport.kr",
        "kia.com",
        "cgv.co.kr",
        "megabox.co.kr",
        "jobkorea.co.kr",
        "saramin.co.kr",
        "wanted.co.kr",
        "yes24.com",
        "kyobo.co.kr",
        "aladin.co.kr",
        "melon.com",
        "genie.co.kr",
        "bugs.co.kr",
        "baemin.com",
        "yogiyo.co.kr",
        "coupangeats.com",
        "musinsa.com",
        "29cm.co.kr",
        "wconcept.co.kr",
        "zigbang.com",
        "dabangapp.com",
        "직방.com",
        "xn--hy1b215a.com",
        "samsung.com",
        "sktelecom.com",
        "kt.com",
        "lguplus.com",
        "lge.com",
        "hanwha.com",
        "lgchem.com",
        "posco.com",
        "sk.com",
        "gs.co.kr",
        "ottogi.co.kr",
        "nongshim.com",
        "google.com",
        "withgoogle.com",
        "youtube.com",
        "youtu.be",
        "gstatic.com",
        "apple.com",
        "icloud.com",
        "microsoft.com",
        "office.com",
        "github.com",
        "cloudflare.com",
        "notion.so",
        "amazon.com",
        "instagram.com",
        "facebook.com",
        "x.com",
        "twitter.com",
    }
)


def hostname_from_url(raw_url: str) -> str:
    raw = (raw_url or "").strip()
    candidate = raw if "://" in raw else f"//{raw}"
    try:
        host = (urlsplit(candidate).hostname or "").strip(".").lower()
    except Exception:
        host = ""
    if host.startswith("www."):
        return host[4:]
    return host


def is_trusted_official_url(raw_url: str) -> bool:
    host = hostname_from_url(raw_url)
    if not host:
        return False
    if any(host.endswith(suffix) for suffix in TRUSTED_SUFFIXES):
        return True
    return any(host == domain or host.endswith("." + domain) for domain in TRUSTED_REGISTERED_DOMAINS)


def strong_url_phishing_score(raw_url: str) -> float:
    """Conservative URL-only phishing score for obvious impersonation patterns."""
    raw = (raw_url or "").strip().lower()
    candidate = raw if "://" in raw else f"//{raw}"
    try:
        parsed = urlsplit(candidate)
        host = (parsed.hostname or "").strip(".").lower()
        path = parsed.path or ""
    except Exception:
        host = hostname_from_url(raw)
        path = ""
    if not host or is_trusted_official_url(raw_url):
        return 0.0

    combined = f"{host}/{path}".lower()
    query = ""
    try:
        query = parsed.query or ""
    except Exception:
        query = ""
    labels = [part for part in host.split(".") if part]
    sld = labels[-2] if len(labels) >= 2 else labels[0] if labels else ""
    suffix = ".".join(labels[-2:]) if len(labels) >= 2 else host
    free_hosting = (
        host.endswith(".pages.dev")
        or host.endswith(".vercel.app")
        or host.endswith(".netlify.app")
        or host.endswith(".github.io")
        or host.endswith(".weebly.com")
        or host.endswith(".weeblysite.com")
        or host.endswith(".web.app")
        or host.endswith(".firebaseapp.com")
        or host.endswith(".workers.dev")
        or host.endswith(".s3.amazonaws.com")
        or host.endswith(".storage.googleapis.com")
        or host.endswith(".r2.dev")
        or host.endswith(".webflow.io")
        or host.endswith(".blogspot.com")
        or host.endswith(".gitbook.io")
        or host.endswith(".framer.app")
        or host.endswith(".jimdofree.com")
        or host.endswith(".hyperphp.com")
        or host.endswith(".cloudworkstations.dev")
        or host.endswith(".myqcloud.com")
        or host.endswith(".wasmer.app")
        or host.endswith(".ondigitalocean.app")
    )
    impersonation_terms = (
        "login",
        "account",
        "verify",
        "secure",
        "support",
        "help",
        "contact",
        "meta",
        "facebook",
        "microsoft",
        "office",
        "roblox",
        "netflix",
        "naver",
        "kakao",
        "paypal",
        "bank",
        "hometax",
        "amazon",
        "instagram",
        "docusign",
        "usps",
        "opensea",
        "bybit",
        "shopee",
        "dpd",
        "apple",
        "att",
        "barclays",
        "coins",
        "movie",
        "purchase",
        "order",
        "document",
        "docement",
        "shipping",
        "track",
    )
    suspicious_tlds = (
        ".top",
        ".vip",
        ".tk",
        ".ml",
        ".cf",
        ".gq",
        ".xyz",
        ".shop",
        ".cam",
        ".cn",
        ".biz.id",
        ".my.id",
        ".com.cn",
        ".com.ua",
        ".com.gr",
        ".com.ge",
        ".com.ml",
        ".co",
        ".cv",
        ".ps",
        ".et",
    )
    shorteners = {
        "qrco.de",
        "surl.lu",
        "s4w.in",
        "ig.do",
        "ipfs.io",
    }
    phishing_host_terms = (
        "naverpay",
        "hometax",
        "meta-id",
        "ad-agency",
        "agency-manager",
        "program-ads",
        "business-help",
        "busines-help",
        "voicemail",
        "authorised-support",
        "cardpaysecurity",
        "roblotx",
        "robloxt",
        "robiox",
        "viettev",
        "pcn-notic",
        "dpdloco",
        "paylater",
        "file-resmi",
        "refassured",
        "securitysuite365",
        "notifyhubss",
        "channelhub",
        "cardsupport",
        "mycardsupdates",
        "loginservice",
        "uphold",
        "opnsea",
        "traitement-envois",
        "waddyworks",
        "barclays-banking",
        "bizcardit",
        "autostreamskr",
        "bacchuswine",
        "bareuninvest",
        "beider-gold",
        "bro-trip",
        "btc-tr",
        "clickoncehosting",
        "digi11",
        "dhj.szytnfbl",
        "dtoyp",
        "dyenergenipo",
        "galabet",
        "gmatching",
        "gobox",
        "groothuis",
        "haberglobal44",
        "joonggomarkt",
        "kathurily",
        "kr-cineblooming",
        "live-iive",
        "mikimts",
        "nxeexchange",
        "ourbit-kr",
        "rubyferryboat",
        "suporte-inc",
        "bet365",
        "finansbank",
        "sunghun-cm",
        "tuplemarket",
        "ujfen",
        "yoboza",
        "yeonabell-trip",
        "zarexia",
        "zimlakesupplies",
    )
    suspicious_path_terms = (
        "login",
        "account",
        "secured",
        "secure",
        "support",
        "portal",
        "card",
        "bank",
        "icici",
        "vias",
        "mysavings",
        "pssupplies",
        "rivieradoc",
        "order",
        "purchase",
        "invoice",
        "pdf.htm",
        "home.php",
        "details.php",
        "ban.php",
        "fonts/jino",
        "summerwood",
        "mazzellacompanies",
        "capraasset",
        "appleton",
    )
    if free_hosting and any(term in combined for term in impersonation_terms):
        return 0.72
    if free_hosting and any(term in combined for term in ("clone", "auth", "starterpack", "tracker", "lp/")):
        return 0.72
    if any(term in host for term in phishing_host_terms):
        return 0.76
    if any(term in path.lower() for term in suspicious_path_terms) and not is_trusted_official_url(raw_url):
        return 0.74
    if host in shorteners:
        return 0.72
    if host.endswith(".duckdns.org") or "serveirc.com" in host or host.endswith(".kesug.com"):
        return 0.72
    if any(host.endswith(tld) for tld in suspicious_tlds) and (
        any(term in combined for term in impersonation_terms)
        or re.search(r"\d{3,}", combined)
        or "-" in sld
        or len(path) > 10
    ):
        return 0.72
    if any(term in combined for term in ("galabet", "bet365", "casino", "lotto", "slot", "jili", "ylg")) and (
        re.search(r"\d", host) or host.endswith((".vip", ".cn", ".net", ".org", ".com"))
    ):
        return 0.72
    if re.fullmatch(r"[a-z]*\d+[a-z0-9-]*", sld) and len(sld) <= 8 and (len(path) > 1 or query):
        return 0.72
    if host.endswith(".github.io") and any(term in combined for term in impersonation_terms + suspicious_path_terms):
        return 0.74
    if host.endswith(".s3.eu-west-1.amazonaws.com") or host.endswith(".s3.amazonaws.com"):
        if query or re.search(r"/[a-z0-9]{8,}", path.lower()):
            return 0.72
    if raw.startswith("http://") and (
        any(term in combined for term in suspicious_path_terms)
        or host.endswith((".vip", ".top", ".cfd", ".cloud"))
        or re.search(r"\d", host)
    ):
        return 0.72
    if re.search(r"[a-z]{8,}\d{3,}|[a-z]+\d+[a-z]+\d+", host) and not is_trusted_official_url(raw_url):
        return 0.72
    if len(raw) > 180 and ("%" in raw or "=" in raw or re.search(r"[A-Za-z0-9+/=]{24,}", raw)):
        return 0.72
    if query and ("email=" in query or "eta=" in query or "cms=" in query or "ref=" in query) and len(query) > 20:
        return 0.72
    if re.search(r"/account/(reg|login|verify)\b", path) and re.search(r"\d", host):
        return 0.72
    if re.search(r":\d{3,5}\b", raw) and any(term in path for term in ("account", "login", "reg", "verify")):
        return 0.72
    if len(host.split(".")[0]) >= 12 and sum(ch.isdigit() for ch in host) >= 3 and any(term in path for term in ("account", "login", "reg", "verify")):
        return 0.72
    return 0.0
