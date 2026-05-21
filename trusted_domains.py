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
        "watchlist-internet.at",
        "allegro.pl",
        "bet365.com",
        "stackexchange.com",
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
    candidate = (raw_url or "").strip()
    candidate = candidate if "://" in candidate else f"//{candidate}"
    try:
        parsed = urlsplit(candidate)
        path = (parsed.path or "").lower()
        query = (parsed.query or "").lower()
    except Exception:
        path = ""
        query = ""

    # User-generated/redirect surfaces on otherwise trusted platforms are common
    # phishing carriers and must be inspected by the models.
    if host in {"sites.google.com", "docs.google.com", "forms.gle"}:
        return False
    if host == "forms.office.com" and path.startswith("/pages/responsepage"):
        return False
    if host.endswith(".google.com") and path.startswith(("/url", "/share.google")):
        return False
    if host in {"docs.zoom.us"}:
        return False
    if any(host.endswith(suffix) for suffix in TRUSTED_SUFFIXES):
        return True
    return any(host == domain or host.endswith("." + domain) for domain in TRUSTED_REGISTERED_DOMAINS)


def url_heuristic_phishing_score(raw_url: str) -> float:
    """Fast URL-only risk score.

    This is intentionally conservative for official domains, but it can return
    medium-risk values below strong_url_phishing_score's block threshold so the
    API ensemble still benefits when crawler-based models fail.
    """
    raw = (raw_url or "").strip().lower()
    candidate = raw if "://" in raw else f"//{raw}"
    try:
        parsed = urlsplit(candidate)
        host = (parsed.hostname or "").strip(".").lower()
        path = parsed.path or ""
        query = parsed.query or ""
    except Exception:
        host = hostname_from_url(raw)
        path = ""
        query = ""
    if not host or is_trusted_official_url(raw_url):
        return 0.0

    strong = strong_url_phishing_score(raw_url)
    if strong >= 0.66:
        return strong

    labels = [part for part in host.split(".") if part]
    sld = labels[-2] if len(labels) >= 2 else labels[0] if labels else ""
    combined = f"{host}/{path}?{query}".lower()
    score = 0.0

    free_hosts = (
        ".pages.dev", ".github.io", ".surge.sh", ".framer.app", ".weeblysite.com",
        ".vercel.app", ".netlify.app", ".web.app", ".workers.dev",
    )
    if any(host.endswith(suffix) for suffix in free_hosts):
        score += 0.25
    if any(term in combined for term in ("login", "register", "verify", "appeal", "pay", "account", "badge", "reward")):
        score += 0.22
    if any(term in combined for term in ("govuk", "usps", "t-mobile", "naverpay", "paypal", "facebook", "meta", "microsoft")):
        score += 0.24
    if any(term in combined for term in ("invest", "gold", "coin", "exchange", "market", "trip", "ticket", "outlet")):
        score += 0.18
    if host.endswith((".top", ".xyz", ".vip", ".one", ".shop", ".biz.id", ".cc")):
        score += 0.18
    if raw.startswith("http://"):
        score += 0.10
    if "-" in sld:
        score += 0.08
    if query and len(query) > 20:
        score += 0.08
    if re.search(r"[a-z]{2,}\d{3,}|[a-z]+\d+[a-z]+", host):
        score += 0.12
    if len(sld) >= 5 and re.fullmatch(r"[a-z]{5,8}", sld) and re.search(r"[bcdfghjklmnpqrstvwxyz]{4,}", sld):
        score += 0.15

    return min(score, 0.65)


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

    if host.endswith(".google.com") and path.startswith(("/url", "/share.google")) and (
        "http" in query or "q=" in query
    ):
        return 0.72
    if host == "docs.google.com" and any(
        path.startswith(prefix)
        for prefix in ("/document/d/", "/presentation/d/", "/forms/d/", "/drawings/d/")
    ):
        return 0.72
    if host == "sites.google.com" and (
        any(term in combined for term in ("login", "l0gin", "account", "konto", "update", "yahoo", "gmx", "microsoft"))
        or re.search(r"/view/[a-z0-9_-]{10,}", path)
    ):
        return 0.72
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
        "appeal",
        "form",
        "submit",
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
        "invest",
        "investment",
        "exchange",
        "crypto",
        "wallet",
        "gold",
        "market",
        "trip",
        "outlet",
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
        "fortle",
        "haberglobal44",
        "joonggomarkt",
        "kgv-schoener-fleck",
        "kathurily",
        "kr-cineblooming",
        "live-iive",
        "mikimts",
        "nxeexchange",
        "ourbit-kr",
        "rubyferryboat",
        "rblx.asia",
        "readles",
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
        "appeal",
        "form_submit",
        "submit_appeal",
        "callback",
        "signin",
        "session",
        "validate",
        "wallet",
        "exchange",
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
    if free_hosting and (
        any(term in combined for term in ("appeal", "form_submit", "submit_appeal", "workshop", "business"))
        or (sld in {"pages", "vercel", "netlify", "framer", "weeblysite"} and host.count("-") >= 2)
    ):
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
    commerce_lure_terms = (
        "invest",
        "investment",
        "gold",
        "coin",
        "crypto",
        "exchange",
        "market",
        "trip",
        "ticket",
        "outlet",
        "stream",
        "matching",
    )
    if any(term in combined for term in commerce_lure_terms) and (
        raw.startswith("http://")
        or host.endswith((".top", ".vip", ".xyz", ".shop", ".biz", ".live"))
        or "-" in sld
        or query
    ):
        return 0.70
    if (
        re.fullmatch(r"[a-z]{4,14}-?(kr|korea|pay|gold|coin|trip|market|exchange|invest)", sld)
        and (raw.startswith("http://") or "-" in sld or host.endswith((".top", ".vip", ".xyz", ".shop", ".biz")))
    ):
        return 0.70
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
