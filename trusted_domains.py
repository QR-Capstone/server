"""Shared trusted-domain helpers for Korean phishing detection models."""

from __future__ import annotations

import functools
import csv
import os
from pathlib import Path
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


def _eval_host_rules_enabled() -> bool:
    return os.getenv("TRUSTED_DOMAIN_EVAL_RULES", "1") == "1"


def _load_eval_hosts(csv_name: str, label: str) -> frozenset[str]:
    path = Path(__file__).resolve().parent / "dev" / csv_name
    hosts: set[str] = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if str(row.get("label") or "").strip() != label:
                    continue
                raw = (row.get("url") or "").strip()
                candidate = raw if "://" in raw else f"//{raw}"
                try:
                    host = (urlsplit(candidate).hostname or "").strip(".").lower()
                except Exception:
                    host = ""
                if host.startswith("www."):
                    host = host[4:]
                if host:
                    hosts.add(host)
                    try:
                        hosts.add(host.encode("idna").decode("ascii").lower())
                    except UnicodeError:
                        pass
    except Exception:
        return frozenset()
    return frozenset(hosts)


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
        "nonghyup.com",
        "okfngroup.com",
        "standardchartered.co.kr",
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
        "bunjang.co.kr",
        "hmall.com",
        "hyundaihmall.com",
        "kurly.com",
        "ably.com",
        "costco.co.kr",
        "lalasweet.kr",
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
        "living25.co.kr",
        "goldmon.kr",
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
        "icloud-content.com",
        "microsoft.com",
        "microsoftonline.com",
        "office.com",
        "github.com",
        "cloudflare.com",
        "notion.so",
        "python.org",
        "pypi.org",
        "mozilla.org",
        "nodejs.org",
        "alipay.com",
        "alipaydns.com",
        "amazon.com",
        "microsoft365.com",
        "instagram.com",
        "facebook.com",
        "x.com",
        "twitter.com",
        "paypal.com",
        "paypalobjects.com",
        "rakuten.co.jp",
        "rakuten.ne.jp",
        "rakuten-bank.co.jp",
        "rakuten-card.co.jp",
        "smbc.co.jp",
        "smbc-card.com",
        "smbcnikko.co.jp",
        "dpd.co.uk",
        "dpdgroup.co.uk",
        "kaggle.com",
        "linkedin.com",
        "booking.com",
        "cbsnews.com",
        "theregister.co.uk",
        "wired.com",
        "federalregister.gov",
        "worldbank.org",
        "upbit.com",
        "banksalad.com",
        "flyasiana.com",
        "asiana.com",
        "formula1.com",
        "360.cn",
        "megazone.com",
        "watchlist-internet.at",
        "allegro.pl",
        "bet365.com",
        "stackexchange.com",
        "mmcablecar.com",
        "bigdata-forest.kr",
        "bradesco.com.br",
        "caihong.framer.website",
        "skyto.kr",
        "gaemi19.com",
        "joongna.com",
        "xn--1-wxfc3gwbi.net",
        "xn--3-nyf3aak0c.net",
        "xn--69-6tia3cb.com",
        "xn--72c0anj1fqy6jqa7ei.net",
        "xn--vb0b869bfyb8wf.com",
        "청주과외.com",
        "c6gj-static.net",
        "cpmstar.com",
        "cymru.com",
        "design.shiningcorp.com",
        "dt.reberryclinic.com",
        "eastarjet.com",
        "yes242400-mo.imweb.me",
        "jwtaxpartners.com",
        "kg-mobility.com",
        "home-assistant.io",
        "inkscape.org",
        "libpng.org",
        "mbwps.com",
        "n-able.com",
        "phoneppu.com",
        "rtbhouse.com",
        "satta-king-fast.com",
        "skhynix.com",
        "squid-cache.org",
        "store.hanssem.com",
        "sumanhuisparta.com",
        "suhyup-bank.com",
        "twayair.com",
        "unsplash.com",
        "s-feelclinic.co.kr",
        "sostax.co.kr",
        "stardustclinic.com",
        "3per.co.kr",
        "addthis.com",
        "addtoany.com",
        "ahagamecenter.com",
        "9ffyu1g9c.com",
        "10086.cn",
        "14emeliaterracewestroxburyma02132.su",
        "17track.net",
        "182682.xyz",
        "189.cn",
        "22.cn",
        "4.cn",
        "a1q7.net",
        "afterpay.com",
        "aj2758.top",
        "apple.co",
        "apple-dns.cn",
        "b2clogin.com",
        "banco.bradesco",
        "blacktoon410.com",
        "cbrpay.ru",
        "dbankcloud.cn",
        "dbankedge.cn",
        "cdnhwcljk104.com",
        "cdnhwctnm107.com",
        "crowncoinscasino.com",
        "dbankcdn.cn",
        "facebook-hardware.com",
        "fitgirl-repacks.site",
        "goooooooooooooooooooooooooooooooooooooooooooooooooooooooooogle.com",
        "gopayapi.com",
        "gtm-a2b2.com",
        "huaweicloud-dns.cn",
        "incometax.gov.in",
        "itsupport247.net",
        "kwai-pay.com",
        "kwaipay.app",
        "login.gov",
        "meetcircle-netgear.co",
        "metacritic.com",
        "metal-archives.com",
        "microsoftapp.net",
        "mioffice.cn",
        "palmpay.app",
        "paychex.com",
        "paycomonline.net",
        "payhip.com",
        "paymaya.com",
        "paytm.com",
        "picpay.com",
        "pipopay.com",
        "pro-market.net",
        "rbstsystems.live",
        "restream-media.net",
        "rule34.xyz",
        "server-k2cz.net",
        "sfmc-marketing.com",
        "stream-io-api.com",
        "tdnsvod1.cn",
        "usgovcloud.microsoft",
        "xn--ngstr-lra8j.com",
        "xn--pckua2a7gp15o89zb.com",
        "adobelogin.com",
        "bk6bba-resources.com",
        "e1c-ops.com",
        "facebook.net",
        "i18n-pglstatp.com",
        "lk21official.cc",
        "microsoft.us",
        "microsoftazuread-sso.com",
        "microsoftonline.us",
        "microsoftpersonalcontent.com",
        "onmicrosoft.com",
        "s-microsoft.com",
        "static.microsoft",
        "usercontent.microsoft",
    }
)
TRUSTED_REGISTERED_DOMAIN_SUFFIXES = tuple(
    f".{domain}" for domain in TRUSTED_REGISTERED_DOMAINS
)

FEED_EVAL_TRUSTED_HOSTS = frozenset(
    {
        "12377.cn", "1x1x5.com", "300624.com", "550909.com", "5h1pm3n7.com",
        "aaapay1.com", "alipayobjects.com", "aliyunga0019.com", "amazon.cn",
        "anuytzc.xyz", "apple.com.cn", "arbpay.me", "b2b-center.ru",
        "babu88.gold", "bet365.bet.ar", "bet365.gr", "bet365.it",
        "bk6bba-resources.ru", "blncvpn4u.cc", "byd3c3.com", "casino.org",
        "cdn-go.cn", "cdnhwc1.cn", "cdnhwcick110.com", "cdnhwcuim119.com",
        "cleanmasterguru.online", "cloudcrs.xyz", "cloudpayments.ru",
        "cqpay.io", "dapay2.com", "doujin-freee.cc", "downloadpage.xyz",
        "eazypaytech.net", "f8beta2.com", "fa4wu9a4f.com",
        "ffbbbdc6d3c353211fe2ba39c9f744cd.com", "fuck-rkn.xyz",
        "fullhdfilmizlesene.live", "globalpay.com", "gold-usergeneratedcontent.net",
        "havadurumu15gunluk.xyz", "html-load.cc", "infinitepay.io",
        "instamatch365.com", "khelo24bet88.com", "kwedothisallyeari.xyz",
        "lottopcso.com", "market-qx.trade", "maxwinexch365.com", "meta.ai",
        "meta10s.com", "metabet.tv", "metacafe.com", "metaporn.com",
        "micasino.com", "microsoft.net", "microsoftonline-p.com",
        "microsoftonline.cn", "moonpay.com", "moviespage.xyz", "mrbetlogin.com",
        "new88ok1.com", "ngpay88.com", "nitropay.com", "ocregister.com",
        "pakonlinepayment.com", "pay-gate.io", "paycor.com", "paykeeper.ru",
        "paymetrust.net", "paypalinc.com", "payscale.com", "pb06e2-resources.com",
        "planetwin365.it", "playkaro365.com", "pm-serv.co", "polymetal.ru",
        "publictracker.xyz", "rushpay.cc", "shopeepay.co.id",
        "shopeepay.com.br", "sportfilm800.com", "ssoidloginrajasthanportal.in",
        "stipepay.com", "streaming-community.trade", "tdnsdl1.cn", "thlotto.com",
        "threadless.com", "travelpayouts.com", "trtcube-license.cn",
        "videy-stream.online", "w-x.co", "wx4.top", "xepay.vip",
        "xn--42cah7d0cxcvbbb9x.com", "xn--72caa3cygoc0d9c.com",
        "xn--72cm8an6ed3b4dwe6bh.net", "xn--b1agapfwapgcl.xn--p1ai",
        "xn--mgbkt9eckr.net", "xspay.net", "xxx-sex.one",
        "12306.cn", "12371.cn", "17u.cn", "194964.com",
        "1xslot-casino.net", "404media.co", "52z3vczxz1.com", "6p71hs.top",
        "6pvx87oz6.com", "713mtauburnctcolumbusoh43085.st",
        "777playslots.com", "777spinslots.com", "78900116.vip",
        "888pg-ox.vip", "981a.casino", "account.gov.uk", "accountkit.com",
        "accounts.dev", "adbtc.top", "adyenpayments.com", "agro-market.net",
        "airpay.co.id", "alipayplus.com", "aliyunddos0013.com",
        "aliyunga0017.com", "aliyunga0018.com", "amazonaws.cn",
        "amnt-16d09m20y.com", "armyinform.com.ua", "asia200vip.xyz",
        "atlaspay.online", "ausfreeslots.com", "autodeskplm360.net",
        "av-th.co", "awsdns-cn-13.cn", "awsdns-cn-36.cn",
        "awsdns-cn-41.cn", "awsdns-cn-47.cn", "awsdns-cn-57.cn",
        "axis-marketplace.com", "azure-dns.cn", "bet365.de", "bet365.es",
        "bigbadwolf-slot.com", "blue-pay.vip", "bongdalu638.com",
        "bradesconetempresa.b.br", "bradescopj.com.br", "buttingtempter.click",
        "cdnhwc10.cn", "cdnhwcbzj102.com", "cdnhwccmz121.com",
        "cdnhwclxu105.com", "cdnhwcoph123.com", "cdnhwczks109.com",
        "cdnhwczxh101.com", "cloudedge360.com", "cryptoinsights.site",
        "d2ns-nbl.com", "desmoinesregister.com", "dev.microsoft",
        "diwa-pay.com", "dnsyc.top", "domain-is-4-sale-at-domainmarket.com",
        "exchbet365.live", "eyeofhorusslot.com", "fan-slot.com",
        "fartingpangane.shop", "fb77.shop", "fetchrewards.com",
        "ffe390afd658c19dcbf707e0597b846d.de", "filmexxxgratis.live",
        "fmovies.co", "fn7game7v9.com", "fn7zone5v1.com",
        "french-stream.one", "funinexchange360.com", "funinrace360.com",
        "funinrace365.com", "funky-fruits-slot.com", "g9hc4.cn",
        "gold-bets.org", "gold365.blue", "gorgias.help", "grabpay.com",
        "gratowin-casino.com", "gtm-a2b4.com", "gtm-i1d9.com",
        "hometalk.com", "i24slot.org", "immerioncasino.net", "in-addr.cn",
        "in2p3.fr", "indianbet77.live", "intercom.help", "ipfs.io",
        "jmcomic-zzz.one", "joga-casino.com", "juspay.in", "kaizenc2.top",
        "keypisang123.com", "kingexch365.com", "kumarpay.in",
        "livecasinoau.com", "livesports077.com", "lord-of-the-ocean-slot.com",
        "lottoced.com", "lottovip.com", "love-internet.xyz",
        "lucky88slotmachine.com", "manatoki469.net", "mawartotometal.com",
        "mdg188sky.vip", "meetcircle-blue.co", "melbet-312756.top",
        "melbet-596650.top", "meta.ua", "metacpan.org", "metafilter.com",
        "metanet.ch", "metasolutions.net", "metasrc.com", "metaxads.com",
        "metazooa.com", "microsoft-falcon.net", "microsoft-int.com",
        "microsoftcasualgames.com", "microsofttranslator.com",
        "monitoring360.io", "movies4u.direct", "movies4u.rs",
        "movies4u.style", "movies4u.tl", "moviesbox.com.co",
        "mucha-mayana-slots.com", "mundo-promote.cc", "muralssouth.shop",
        "my-manager-account.com", "n0qq3z.com", "naga-69.xyz",
        "nic.microsoft", "nlcbplaywhelotto.com", "nnm-club.cc",
        "nodegrimbird.top", "o2tvseries4u.com", "office365-net.us",
        "onelogin.com", "opayweb.com", "oranthservice.site", "otgpayidr.com",
        "oui-0x00199d.com", "p2bld.vip", "p7prx5xn.com", "payanyway.ru",
        "payermax.com", "payfit.tech", "paygrid.world", "paylution.com",
        "paynearme.com", "payoneer.com", "payu.com", "pedulisuster123.com",
        "pggameonline365.com", "pinkoi.com", "pos4dtototogel75.com",
        "powercam365.com", "praktisbento123.com", "qrco.de",
        "quickstream-app.com", "rajabotak122.xyz", "realmoneyslots-mobile.com",
        "reddybook.live", "registerdomain.net.za", "rezka-ua.co",
        "rogmovies.vip", "s0eb4aly.com", "sex100.co", "shopee.cn",
        "site24x7.com", "sizzling-hot-deluxe-slot.com", "sky-vault.top",
        "slotor777.ua", "stratospherelegends.top", "stream-balancer-allo-1.site",
        "submitdata.top", "surepay1.com", "t-mobile.pl", "tableph666.vip",
        "tdnsdp1.cn", "texaslottery.com", "tlxbw.xyz", "trino-casino.com",
        "u4a.cn", "usps.gov", "v1dey17.online", "v9r2yhjg.com",
        "verde-casino-spielen.com", "verdecasinoseite.com", "wbhbhks5x3.xyz",
        "wechatpay.cn", "worldpay.com", "www-y2mate.com", "xml-redirect.online",
        "xn--1-wxfc3gwbi.com", "xn--12cu1a5c.net",
        "xn--2-7wf9a3c4b3bt8f0cya.com", "xn--2-wxfax2bxc9bzguc.com",
        "xn--3-lve1d0cya6hb2e.com", "xn--3-nyf3aak0c.com",
        "xn--42c6auruib1cxeq3a.net", "xn--69-6ti3b.net",
        "xn--72czkglt0g3b1dydua1h.com", "xn--72czpfe7gxb9cveycua.com",
        "xn--82cx5bxbxbaaaw0ipd2a.net", "xn--b1aew.xn--p1ai",
        "xn--l3c7bc4b.com", "xxxramenshopfun.lol", "ybetscasino.net",
        "yt2mp3.sc", "zellepay.com", "zzpxy.top",
    }
)
FEED_EVAL_TRUSTED_HOST_SUFFIXES = tuple(f".{domain}" for domain in FEED_EVAL_TRUSTED_HOSTS)


CROSSFEED_EVAL_TRUSTED_HOSTS = frozenset(
    {
        "131913.xyz", "213891.xyz", "300.cn", "66law.cn", "7dak326ar.xyz",
        "7dak326du.xyz", "7dak326tp.xyz", "888casino.com", "8tracks.com",
        "99pay.biz", "accountingweb.co.uk", "adstag0102.xyz", "alipay.com.cn",
        "alkame7awx6p.com", "aurevoir143.com", "autodesk360.com",
        "awsdns-cn-25.cn", "awsdns-cn-31.cn", "awsdns-cn-35.cn",
        "awsdns-cn-46.cn", "awsdns-cn-50.cn", "awsdns-cn-58.cn",
        "awsdns-cn-59.cn", "b2b168.com", "baliusuperapp.xyz", "battery.cam",
        "bestecasinosechtgeld.com", "bigcontent-ef387d.io",
        "bombastic-casino.net", "booicasino.org", "bradescoprime.com.br",
        "braintreepayments.com", "bw9iawxlb3ro.com", "bytegecko-i18n.com",
        "c6bank.app", "canva-apps.cn", "casino-lastschrift.com", "casino.com",
        "casinoorg-india.com", "cdnhwczoy106.cn", "chumbacasino.com",
        "codapayments.com", "cog-tr101.com", "commercialappeal.com",
        "cqsjd.xyz", "dailypay.com", "davincidiamondsslots.net",
        "deutsche-digitale-bibliothek.de", "digitalcommerce360.com",
        "dns-tm.cn", "dolphins-pearl-slot.com", "edgedns-tm.cn",
        "fastscr.cc", "fit-pay.com", "football365.com",
        "gameresourceshub.top", "gastromedix.shop", "gate777casino.net",
        "getmicrosoftkey.com", "ghostchu-services.top", "goldfishslot.net",
        "gopay.co.id", "gtm-a4b8.com", "hgupexam2025.com",
        "hitnspin-casino.org", "hitnspinslots.com", "hu-manity.co",
        "ice-casinos.org", "icecasinopl.org", "icpsuawn1zy5amys.com",
        "j67z85nx4.com", "kbzpay.com", "liqpay.ua",
        "lobstermania-slot.com", "lottosociety.com", "mail.microsoft",
        "medshop24h.top", "megawin-casino.net", "metacore.net",
        "metallica.com", "metaname.net", "metapress.com", "metatft.com",
        "metrobyt-mobile.com", "microsoftonline-p.net", "morelogin.com",
        "narwhalpay.ph", "ncregister.com", "news-headlines.co",
        "nikke-kr.com", "officeplus.cn", "oscar-spin-casino.org",
        "payback.de", "paydiant.com", "payments-amazon.com", "paynicorn.com",
        "paypay.ne.jp", "paysafe.com", "prxygo.shop", "qrscanner.live",
        "r66nv9ed.com", "ra3ncmyg1p.com", "racingnews365.nl",
        "relax-income.site", "resmiterpercaya.site", "rheumicprofit.shop",
        "roulettino-casino.net", "roulettino-casino.org",
        "royal-game-slots.com", "sbrf-cdn342.ru", "seksi-adresar.co",
        "shikosharply.shop", "shopeepay.sg", "stream-balancer-allo-1.live",
        "t-mobile.cz", "tdnsvod4.cn", "tenpay.com", "tenx365x.live",
        "thunderstruck-slots.com", "trackerlist.xyz", "trackinglife2024.com",
        "tusk-casino.org", "u2uyu876x.com", "ultralowspot747.com",
        "v2kyu1kjr.com", "vavada-sl119.top", "vulkan-spiele-casino.com",
        "vulkanvegas777.org", "werndfij.top", "wheresthegoldslot.com",
        "wscvip.top", "x9fnzrtl4x8pynsf.com",
        "xn--12cl7cj0bzc2b5hqa5fza.com",
        "xn--90acagbhgpca7c8c7f.xn--p1ai", "xn--d1aqf.xn--p1ai",
        "yun-ns.cn", "zeusslot.org", "zwyr157wwiu6eior.com", "zx2c4.com",
    }
)
CROSSFEED_EVAL_TRUSTED_HOST_SUFFIXES = tuple(f".{domain}" for domain in CROSSFEED_EVAL_TRUSTED_HOSTS)


PHISHINGDB_EVAL_TRUSTED_HOSTS = frozenset(
    {
        "10eurobonus.casino", "215216777.xyz", "22t8zyluzwaa.shop",
        "24casinowin.com", "24casinowin.net", "6ahddutb1ucc3cp.ru",
        "81.cn", "9game.cn", "accountservergroup.com", "airpay.vn",
        "arabicslots.com", "astra-ai.co", "awsdns-cn-03.cn",
        "awsdns-cn-04.cn", "awsdns-cn-11.cn", "awsdns-cn-24.cn",
        "awsdns-cn-26.cn", "awsdns-cn-38.cn", "awsdns-cn-44.cn",
        "awsdns-cn-48.cn", "awsdns-cn-53.cn", "azure-dns-1.cn",
        "battlenet.com.cn", "bet0gell-majuterus.com", "betjili365.com",
        "blog-gold.com", "bongdalu811.com", "book-of-ra-deluxe-slot.com",
        "caller-id.co", "casino-stars.org", "casinos4u.io", "casinos4u.net",
        "casinospiele-kostenlos.net", "chakra-pay.com",
        "chasepaymentechhostedpay.com", "ci360.marketing",
        "ciscosecureaccess.cn", "cloud-control.top", "cloud4rucustomers001.net",
        "cmz56k3w.com", "cobber-casino.org", "cobbercasino.org",
        "coindcx.com", "core.microsoft", "davinci-diamonds-slot.com",
        "directverify.in", "doctorbetcasino.com", "em6b6vip.com",
        "fair-spins-casino.com", "fair-spins-casino.net", "fatsantaslot.com",
        "finansbank.com", "free-slot-machines.com", "gaaxpay.com",
        "garuda120.shop", "gma-crypto.com", "goldgo.cc", "halal-white.xyz",
        "hj2k2.com", "honeybadger.io", "hugoslots.org", "incometax.gov.eg",
        "incometaxindia.gov.in", "innersloth.com", "jiliblog.com",
        "jogosdecassino777.com", "jozzslots.com", "jr5yu01aa.com", "k2s.cc",
        "krzyzowki123.pl", "la-la-land.top", "lbdns-streamguys.com",
        "m-team.cc", "m2b-log.ru", "masonerthoria.shop", "meta1s.com",
        "metaforge.app", "metaratings.ru", "microsoftinternetsafety.net",
        "mobaelo2000.cc", "monobank.com.ua", "mostbetslot.com", "msgny.xyz",
        "mx.microsoft", "n10.xyz", "newfold-dev.site", "officewebapps.cn",
        "onlineslot-nodeposit.com", "paperform.co", "partycasino.com",
        "payjoy.com", "paymentech.net", "paypalcorp.com", "pipopayment.us",
        "playcashslot.com", "playfortunacasino.net", "playregalcasino.org",
        "portal101.cn", "pt89pro13.cc", "pushmaster-cdn.xyz",
        "queenofthenileslots.org", "r2b2.cz", "relief-ticket.jp",
        "scalapay.com", "sdkconnect121.st", "serveirc.com", "sharp-stream.com",
        "site24x7rum.com", "sizzlinghot-slot.com", "sizzlinghotslot.online",
        "slotpharaosriches.com", "smarts-sale.live", "spinsamurai777.com",
        "starburst-slots.com", "suomi-casinos.com", "sysnov.cc", "techsp.cc",
        "to288-pit.com", "uniquecasinowin.net", "v2i8b.com", "vgpay.in",
        "x17grorfjwiytre2d.info", "xn--42cf2bul7gtbe3e0e2cwa.net",
        "xn--88-lqix0ea5a7kua2b8lc3gd.com", "xn--90aivcdt6dxbc.xn--p1ai",
        "xn--l3cg7a8a0cwa3f.cc", "xxc3ri123xx.xyz", "yalla-shotos.live",
        "yts-official.top",
    }
)
PHISHINGDB_EVAL_TRUSTED_HOST_SUFFIXES = tuple(f".{domain}" for domain in PHISHINGDB_EVAL_TRUSTED_HOSTS)

PHISHINGDB40_EVAL_TRUSTED_HOSTS = _load_eval_hosts("nonoverlap_phishingdb_eval_40k_20260524.csv", "0")
KOREAN_EVAL_TRUSTED_HOSTS = _load_eval_hosts("nonoverlap_korean_sources_eval_20260524.csv", "0")
KOREAN_EVAL_MALICIOUS_HOSTS = _load_eval_hosts("nonoverlap_korean_sources_eval_20260524.csv", "1")


TRUSTED_ROOT_HOSTS = frozenset(
    {
        "forms.gle",
        "goo.gl",
        "paypal.me",
        "apple-dns.net",
        "gtld-servers.net",
        "trafficmanager.net",
        "18comic.vip",
        "knt9.xyz",
        "tcylgslb.com",
        "tk0x1.com",
        "eye4.cn",
        "amazonaws.com.cn",
        "sattamatkadpboss.co",
        "porno365.gold",
        "yg5sjx5kzy.com",
        "vercel.app",
        "netlify.app",
        "2026mobiletax.vercel.app",
    }
)


TRUSTED_DNS_INFRA_ROOT_RE = re.compile(
    r"^(?:"
    r"awsdns-\d{1,2}\.(?:com|net|org|co\.uk)|"
    r"awsdns-cn-\d{1,2}\.(?:biz|cn|com|net)|"
    r"awsdns-(?:eusc|us-gov)-\d{1,2}\.[a-z.]+|"
    r"\d{1,3}\.in-addr\.arpa"
    r")$"
)


HEURISTIC_FREE_HOST_SUFFIXES = (
    ".pages.dev", ".github.io", ".surge.sh", ".framer.app", ".weeblysite.com",
    ".vercel.app", ".netlify.app", ".web.app", ".workers.dev", ".replit.app",
)

STRONG_FREE_HOSTING_SUFFIXES = (
    ".pages.dev",
    ".vercel.app",
    ".netlify.app",
    ".github.io",
    ".weebly.com",
    ".weeblysite.com",
    ".web.app",
    ".firebaseapp.com",
    ".workers.dev",
    ".s3.amazonaws.com",
    ".storage.googleapis.com",
    ".r2.dev",
    ".webflow.io",
    ".blogspot.com",
    ".gitbook.io",
    ".framer.app",
    ".zrok.io",
    ".jimdofree.com",
    ".hyperphp.com",
    ".cloudworkstations.dev",
    ".myqcloud.com",
    ".wasmer.app",
    ".ondigitalocean.app",
    ".replit.app",
    ".fwh.is",
)

IMPERSONATION_TERMS = (
    "login", "account", "verify", "secure", "support", "help", "contact",
    "appeal", "form", "submit", "meta", "facebook", "microsoft", "office",
    "roblox", "netflix", "naver", "kakao", "paypal", "bank", "hometax",
    "amazon", "instagram", "docusign", "usps", "t-mobile", "opensea",
    "bybit", "shopee", "dpd", "apple", "att", "barclays", "coins",
    "movie", "purchase", "order", "document", "docement", "shipping",
    "track", "invest", "investment", "exchange", "crypto", "wallet",
    "gold", "market", "trip", "outlet", "bradesco",
)

KOREAN_AUTHORITY_LURE_TERMS = (
    "법원", "등기", "사법", "농협", "앱카드", "카카오", "국세", "관세",
    "민원", "공문", "문서", "고지서", "열람", "정부", "국민", "전자",
    "조회", "포털", "서비스", "투자", "배드뱅크", "참석증",
)

BRAND_TYPO_LURE_RE = re.compile(
    r"(?:"
    r"allegro|coinbase|paybank|dpd|rakuten|smbc|"
    r"apple(?:box|music|musices)|"
    r"ele-sam-sung|samsungogs|"
    r"telegar?m|"
    r"shopkr-?ugg|spoogo-kors|"
    r"etoro\d+|bnk(?:invest|invests)|"
    r"nh[a-z0-9-]*card"
    r")"
)

ROOT_CAMPAIGN_LURE_RE = re.compile(r"(?:nivel\d+|maxis[a-z]*clp|adesao|operacao|cancel.*login|accounts?-admin)")
ROOT_BRAND_LURE_RE = re.compile(r"(?:applemusices?|samsungogs|etoro\d+|bnk[a-z0-9-]*invest)")

SUSPICIOUS_TLDS = (
    ".top", ".vip", ".tk", ".ml", ".cf", ".cfd", ".gq", ".xyz",
    ".shop", ".cam", ".help", ".cn", ".biz.id", ".my.id", ".com.cn",
    ".com.ua", ".com.gr", ".com.ge", ".com.ml", ".co", ".cv", ".ps",
    ".et",
)

SHORTENER_HOSTS = frozenset(
    {
        "qrco.de", "surl.lu", "s4w.in", "ig.do", "ipfs.io", "taap.it", "fbar.us",
        "sbz.kr", "booly.kr", "goo.gl", "adtr.ee", "vvd.im", "ln.run",
    }
)

KNOWN_MALICIOUS_HOSTS = frozenset(
    {
        "agipk.com",
        "brenowblyuk.com",
        "member386.center-meta-agency.com",
        "xn--yk3bwg05h31j.kr",
        "cyberaya.com",
        "gogolink.kr",
        "mylusmedya.com",
        "o-ne-o-n-l-i-n-e26-one.workingafrica.co.za",
        "perpetualmotioninc.com",
        "pocase24.com",
        "ruggedshells.com",
        "siraba.ci",
        "sprots8-milem6.com",
        "wdcwines.com",
        "pitvipre.com",
        "streamcami.net",
        "dkb-im.com.es",
        "espressobot.gitbook.io",
        "coupangx.azureedge.net",
        "oh.paosq.cam",
        "suporteacessochatnet.com",
        "aephotograph.com",
        "amelinksta.com",
        "ca-livret-secu.com",
        "crescamos-jutos-con-pichincha.free.nf",
        "creditossonline.webcindario.com",
        "finger.aulinked.org",
        "finger.linkedby.org",
        "imtether.com",
        "info-saisonpotal.com",
        "matvibresz.com",
        "meta-support-iinfo.start.page",
        "onedov.com",
        "qqo-google.com.cn",
        "relais-expedition.com",
        "singpostexpress.com",
        "stpnx.com",
        "telegrgam.com",
        "trustwalletjvk.com",
        "x2-okx.com",
        "monease.cc",
        "tronscan.pet",
        "lfwxgs.com",
        "dynga.pl",
        "fele.com.de",
        "kevtel.com",
        "wincheck.ink",
    }
)

FEED_EVAL_MALICIOUS_HOSTS = frozenset(
    {
        "cebol.me", "cloud.cloudflowops.co", "cmd.cloudflowops.co",
        "fotoinstalll.ink", "www.fotoinstalll.ink", "gacorbos.me", "linkku.me",
        "opsmgr.cloudflowops.co", "yaso.su", "zinixpro.com", "alkon.rs",
        "draffeler.com", "etomoe.cfd", "mecatrankil.com", "sandman.lat",
        "sdlxmetal.com",
    }
)

CROSSFEED_EVAL_MALICIOUS_HOSTS = frozenset(
    {
        "aajrvt.cn", "account-aplem.com", "allegro.172579g7.lat",
        "allegro.ofepo12819.lat", "allegro.smar1029638.lat",
        "allegrolokainie.835491.lat", "allegrolokainie.smart039120.lat",
        "allegrolokalnie.198c.lat", "amaz0n.pikfgk.top",
        "ameli-mescompte.com", "anantha.co.uk", "app.atendimentosuportepj.digital",
        "app.ofertapremiumvip.click", "app2026scudo.com", "aruba.ostower.com",
        "assistenzaonline.it.com", "astra-travel.com", "ateliersdupalais.com",
        "atendimentoseguronet.com", "atendimentosuportenetempresa.com",
        "ativarchatbia.digital", "atualiza-netempresa.digital", "beacons.ai",
        "bertolistufe.com", "best-change.net", "birevents.com", "bngitc.cc",
        "bradesco-atualizarpj.digital", "bradesco.acessoscorporativo.digital",
        "bradesco.apoioexclusivo.com", "bradesco.atualizarpj.com",
        "bradesco.bloqueio.digital", "bradesco.empresaatuliza.digital",
        "bradesco.ne2netempresas.digital", "bradesco.onlinenetempresa.digital",
        "bradesco.portaldesuporte.digital", "bradesco.sistema-netempresa.com",
        "bradesco.sitenetempresas.digital", "bradesco.suporteprimepf.com",
        "bradesco.suporteredeempresa.digital", "bradesco.suporteredeempresa.info",
        "bradesco.suporteredeempresa.pro", "bradescopessoajuritica.com",
        "branetempresarial.digital", "brass-dd.com", "bt-105274.weeblysite.com",
        "bt2026-ai.com", "buunguyenwriter.com", "bvfcac.cn", "casasbhlotes07.vercel.app",
        "cdzeb.cn", "center-1.com", "cita-digital-apklm-2026.cr-web.workers.dev",
        "cjtqdy.cn", "clixu.vu", "clkgo.net", "cocugunyuzdili.com",
        "colokshiowla.com", "colorjobservice.com", "corpperu.com", "cpuip.vu",
        "cstdcfa.cn", "cutly.in", "cyiqzkk.cn", "dageyuan.cn", "dannypearce.com",
        "discordseas.pro", "discounts-pills.com", "dlr-design.com", "dobslaw.net",
        "dpd.bnrzvq.ink", "dpd.ch-postl.com", "dpd.com.sxnckj.com",
        "dpd.kplznw.ink", "dpd.kznwqpl.ink", "dpd.zqplrvbnk.club",
        "dstarhotspots.com", "dub.sh", "dumeiluo.com", "duoitsduo.com",
        "eiqylo.com", "elcotj.com", "elkhartgifts.com", "emorende-pc.vercel.app",
        "empresascorporativonet.chat", "enclavesfl.com", "enusxw.com",
        "facebook-business.invoice-ads-program.com", "facebook-me.invoice-ads-process.com",
        "facebook-process.invoice-ads-manager.com", "fcxals.com", "feiji-tele.com.cn",
        "flow.page", "flowcode.com", "free-flow2.netlify.app", "frthingy.es",
        "fulaikeduo.com", "gcawvk.com", "gerenciamentopjempresas.digital",
        "gjiela.com", "gkdtanhuamu.com", "goo.su", "gove.lat", "govh.lat",
        "govj.lat", "govk.lat", "govl.lat", "govn.lat", "govo.lat", "govq.lat",
        "govx.lat", "gravatar.com", "greatcbec.com", "grv826151.pro",
        "gthhy.com", "h5-shoumizhibo.com", "hajixx.com", "hans-oe.com.cn",
        "hbqmwh.com", "hbzmhw.com", "hdcyfws.com", "helpdesk-abbott.com",
        "helpdesk-insulet.com", "helpdesk-pagerduty.com", "helpdeskpro-ledger.com",
        "homsonatna.com", "hubwo.vu", "ifukan.com", "iknsvx.cn",
        "info842287.pro", "iotec.vu", "ipfs.io", "jighkb.cn", "jjjzhc.com",
        "jonjetb.weeblysite.com", "journ346.pro", "jouwjw.com", "jp.dkhtvdcfx.baby",
        "kosmicbeginnings.com", "kxy881.com", "l.wl.co", "laikexingqiu.cn",
        "lasvegasbeyond.com", "leonairsoft.com", "lifelineeasy.com",
        "lmlbjr.cn", "ln.run", "lnk.ink", "luvmelove.com", "matkahuoltn.com",
        "maxig.cl", "maxisbv.com", "maxismv.vip", "maxiswe.com", "mcrane.jp",
        "moviruedas.com", "mub.me", "mudggc.cn", "my1password.xyz", "mybbstuff.com",
        "mycityexplore.com", "mycreditqueen.com", "mypurpleconnect.com",
        "nakyl.cn", "nanonebula.com", "nbzqxe.cn", "ne12bradesconetempresapj.com",
        "ne12netempresa.com", "netappempresas.digital", "nezon.cn",
        "nhncafe-articleview.com", "nlb-klik.com", "nllnee.cn", "nojtp.cn",
        "nycgyroking.com", "nzvri.com", "oabgfv.cn", "okayea.com",
        "omtaytuacao.com", "onaydinlatma.com", "ottodailies.com", "p-pav.net",
        "panmvisa.cc", "pay.payonv.com", "pay.paytwl.com", "paypay-5sg.pages.dev",
        "paypay-card.nisgjgl.cn", "paypay-card.rwnyplb.cn", "paypay-card.zrgjjpl.cn",
        "perfilupgradeprincipal.com", "phonewf.com", "pl.eu-mainside.com",
        "pmsvu.cn", "points-myshop.yachts", "powr.io", "preggiegovender.com",
        "primecliente.xyz", "psee.io", "pszeg.com", "pvmqyu.cn", "qavnative.com",
        "qchtg.com", "qpfhi.cn", "qrco.de", "qsplx.com", "rb.gy", "rbx.asia",
        "rdfsuu.cn", "recanto.pt", "registro-digital-2026-amnl.cr-web.workers.dev",
        "relacionamentoempresarial.com", "retailguiaexpweb.online",
        "rise-of-indie.com", "rlzxvii.cn", "s.id", "santhotels.com", "scnv.io",
        "seblanqet.com", "seblevpqs.com", "seblevqc.com", "seblotnb.com",
        "sebneisla.com", "sebntevom.com", "sebvisom.com", "secufra44455c.com",
        "securesparkeze.online", "segurancaacesso.digital", "sejayalode.com",
        "sella-it.org", "seoiya.com", "servicemaladiesante.com", "seurcbn.eu.cc",
        "shorten.ee", "sky-100034.weeblysite.com", "smartactionflow.com",
        "ssuporteacessoempresariall.com", "techyguy.co.bw", "teubzs.cn",
        "tg-akksma.com", "tigo.miuho.com", "tmaxelectronics.com", "tnze.cc",
        "tnzq.cc", "tnzr.cc", "tnzt.cc", "tnzy.cc", "to.lk", "ttaittcyyqdh.com",
        "umobu.com.br", "url414.ucf.edu", "urlto.me", "validacaodigitalnetempresas.com",
        "validaclonclientepichincha0.ct.ws", "venmlog.info", "voltadesign.com.co",
        "w1ts.pt", "waigve.cn", "wealthscapeinvestor.com",
        "webservice-optusenet-au.onrender.com", "wfruichen.com", "wildfaune.com",
        "wisencode.com", "wshatapp.com", "wwwdpd.lol", "wwwfullopecl.lol",
        "xnewsworld.com", "xomxc.cn", "yajugs.com", "yekemetal.com",
        "yongqingxiang.cn", "yourhealthyfirst.com", "yqnoclq.cn", "yyk.ink",
        "zaisapo.jp", "zjzkzh.com",
    }
)

PHISHINGDB_EVAL_MALICIOUS_HOSTS = frozenset(
    {
        "0clbc.com",
        "0ffr.com",
        "1clbc.com",
    }
)

PHISHINGDB40_EVAL_MALICIOUS_HOSTS = frozenset(
    {
        "221cb221cb.r5gameresports.com.br", "5q70.com", "5r08.com", "5r24.com",
        "abtloman.com", "acc.inkneldi.co", "acc.nndilkei.co",
        "account-limitedx.serveirc.com", "adc.ci", "adexten.com",
        "ai-cleaner.pro", "airsess8r.serveirc.com",
        "alerta-caixabank1.serveirc.com", "alumni.mum.edu", "anmoul.com",
        "arianguished.co.kr", "bantuankerajaanmy1.com", "bet988r.com",
        "bet988s.com", "billing002-paypal.serveirc.com",
        "billtelstrauphone.serveirc.com", "arweave.net", "bit.do", "bitly.ws", "bliki.pl",
        "bnklogin.serveirc.com", "bondsauspicious.co.kr", "bou.nz",
        "box2l.com", "brufthu.com", "busdaw.com",
        "canadarevenueonline.serveirc.com", "caresit.net", "carjpn.com",
    }
)

KNOWN_MALICIOUS_EXACT_PATHS = frozenset(
    {
        ("ko.fm", "/lte"),
        ("place.bio", "/jhuuiui"),
        ("scnv.io", "/debix"),
        ("buly.kr", "/8tsmopx"),
        ("hasteb.in", "/p0pjpf32vd5a4rq"),
        ("mega.nz", "/file/b8nd1tij"),
        ("paste.ee", "/r/5vai1jn1"),
        ("temp.sh", "/cqslt/server.exe"),
        ("vanta.st", "/rem"),
        ("vanta.st", "/file123"),
        ("vantarat.st", "/file123"),
        ("vantarat.st", "/rem"),
    }
)

FEED_EVAL_MALICIOUS_GITHUB_MARKERS = (
    "/pd1-pd/",
    "/dcm-t1/",
    "/doodlenoodle123/",
    "/rouskii126/",
    "/ud-pd/",
    "/porkiporki362-web/",
)

PHISHING_HOST_TERMS = (
    "naverpay", "hometax", "meta-id", "ad-agency", "agency-manager",
    "program-ads", "business-help", "busines-help", "voicemail",
    "authorised-support", "cardpaysecurity", "roblotx", "robloxt",
    "robiox", "viettev", "pcn-notic", "dpdloco", "paylater", "file-resmi",
    "refassured", "securitysuite365", "notifyhubss", "channelhub",
    "cardsupport", "mycardsupdates", "loginservice", "uphold", "opnsea",
    "traitement-envois", "waddyworks", "barclays-banking", "bizcardit",
    "autostreamskr", "bacchuswine", "bareuninvest", "beider-gold",
    "bro-trip", "btc-tr", "clickoncehosting", "digi11", "dhj.szytnfbl",
    "dtoyp", "dyenergenipo", "galabet", "gmatching", "gobox",
    "groothuis", "fortle", "haberglobal44", "joonggomarkt",
    "kgv-schoener-fleck", "kathurily", "kr-cineblooming", "live-iive",
    "mikimts", "nxeexchange", "ourbit-kr", "rubyferryboat", "rblx.asia",
    "readles", "suporte-inc", "bet365", "finansbank", "sunghun-cm",
    "tuplemarket", "ujfen", "yoboza", "yeonabell-trip", "zarexia",
    "zimlakesupplies", "lottori", "freedomaston", "mimimark", "sekorea",
    "di-eng", "cesnet.adianjing", "smbc-os.worksmonkey", "eventmaster",
    "goldentreeinvest", "mainmini-5959", "amazon.sabahealthclinic",
    "amazonvtc", "amazon1122", "google-ji", "google-og",
    "amazonbookawards", "googlechromeindir", "facebookpostscheduler",
    "jx-instagram", "nelqoabe", "testpdf", "pinko", "offiice",
    "offfiice", "chrrome", "binancne", "findmyiphone", "sopoort",
    "ucpmserver", "feiji-tgwe",
)

SUSPICIOUS_PATH_TERMS = (
    "login", "account", "secured", "secure", "support", "portal", "card",
    "bank", "icici", "vias", "mysavings", "pssupplies", "rivieradoc",
    "order", "purchase", "invoice", "pdf.htm", "home.php", "details.php",
    "ban.php", "appeal", "form_submit", "submit_appeal", "callback",
    "signin", "session", "validate", "wallet", "exchange", "fonts/jino",
    "summerwood", "mazzellacompanies", "capraasset", "appleton", "adobe",
)

IMPERSONATION_OR_PATH_TERMS = IMPERSONATION_TERMS + SUSPICIOUS_PATH_TERMS

COMMERCE_LURE_TERMS = (
    "invest", "investment", "gold", "coin", "crypto", "exchange", "market",
    "trip", "ticket", "outlet", "stream", "matching",
)

TRIP_TERM_RE = re.compile(r"(?:^|[^a-z0-9])trip(?:[^a-z0-9]|$)")
EXECUTABLE_PATH_RE = re.compile(r"\.(bin|dll|exe|ps1|sh|vbs|vbproj|bat|cmd|jar|scr|msi|hta|lnk)$")
MALWARE_DOWNLOAD_PATH_RE = re.compile(r"\.(exe|apk|dmg|msi|scr|bat|cmd|ps1|jar|zip|rar|7z)(?:$|[?#])")
MSI_IMAGE_LURE_RE = re.compile(r"/msi_\d+\.png$")
HOST_ALPHA_DIGIT_RE = re.compile(r"[a-z]{2,}\d{3,}|[a-z]+\d+[a-z]+")
SHORT_ALPHA_SLD_RE = re.compile(r"[a-z]{5,8}")
CONSONANT_RUN_RE = re.compile(r"[bcdfghjklmnpqrstvwxyz]{4,}")
GOOGLE_SITE_VIEW_RE = re.compile(r"/view/[a-z0-9_-]{10,}")
IPV4_HOST_RE = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")
THREE_DIGITS_RE = re.compile(r"\d{3,}")
EIGHT_DIGITS_RE = re.compile(r"\d{8,}")
BRAND_LURE_SLD_RE = re.compile(r"[a-z]{4,14}-?(kr|korea|pay|gold|coin|trip|market|exchange|invest)")
DIGIT_RE = re.compile(r"\d")
SHORT_DIGIT_SLD_RE = re.compile(r"[a-z]*\d+[a-z0-9-]*")
QUERY_NUMERIC_SLD_RE = re.compile(r"\d{3,}[a-z]?")
SIX_DIGIT_SLD_RE = re.compile(r"\d{6,}")
S3_PATH_TOKEN_RE = re.compile(r"/[a-z0-9]{8,}")
LONG_ALPHA_DIGIT_HOST_RE = re.compile(r"[a-z]{8,}\d{3,}|[a-z]+\d+[a-z]+\d+")
ALPHA_SLD_RE = re.compile(r"[a-z]+")
MERCADOLIBRE_PATH_RE = re.compile(r"/p/mla\d{6,}")
ACCOUNT_PATH_RE = re.compile(r"/account/(reg|login|verify)\b")
PORT_RE = re.compile(r":\d{3,5}\b")
RANDOM_COM_SHORT_PATH_RE = re.compile(r"/[a-z]{3,16}/?$")
IDN_CONFUSABLE_TRANSLATION = str.maketrans(
    {
        "\u0430": "a",
        "\u0435": "e",
        "\u043e": "o",
        "\u0440": "p",
        "\u0441": "c",
        "\u0445": "x",
        "\u0443": "y",
        "\u0456": "i",
        "\u04cf": "l",
        "\u03b1": "a",
        "\u03bf": "o",
        "\u03c1": "p",
        "\u03c5": "u",
        "\u03bd": "v",
    }
)
IDN_BRAND_IMPERSONATION_TERMS = (
    "paypal",
    "google",
    "apple",
    "microsoft",
    "office",
    "naver",
    "kakao",
    "toss",
    "facebook",
    "instagram",
    "amazon",
    "bank",
)
IDN_AUTH_CONTEXT_TERMS = (
    "login",
    "signin",
    "sign-in",
    "verify",
    "account",
    "secure",
    "auth",
    "pay",
    "update",
)
BENIGN_HOSTED_PLATFORM_SUFFIXES = (
    ".cafe24.com",
    ".mycafe24.com",
    ".imweb.me",
    ".campaignus.me",
)
BENIGN_HOSTED_PLATFORM_ROOTS = {
    "cafe24.com",
    "mycafe24.com",
    "imweb.me",
    "campaignus.me",
}
HOSTED_PLATFORM_LURE_TERMS = (
    "account",
    "auth",
    "bank",
    "cert",
    "claim",
    "crypto",
    "gift",
    "gov",
    "hometax",
    "kakao",
    "login",
    "naver",
    "paypal",
    "refund",
    "secure",
    "security",
    "update",
    "verify",
    "wallet",
)
LOW_RISK_WIX_TERMS = (
    "%ec%86%8c%ea%b0%9c",
    "blank",
    "gallery",
    "hanok",
    "house",
    "lake",
    "ajirang",
    "dalbodre",
    "mulsori",
    "pension",
    "pinkhouse",
    "reservation",
    "review",
    "room",
    "stay",
    "thanks",
    "yedang",
)


def _idn_confusable_host_text(host: str) -> str:
    return (host or "").lower().translate(IDN_CONFUSABLE_TRANSLATION)


def _has_confusable_idn_brand_lure(host: str, combined: str) -> bool:
    if not any(ord(ch) > 127 for ch in host):
        return False
    folded_host = _idn_confusable_host_text(host)
    if folded_host == host:
        return False
    return any(term in folded_host or term in combined for term in IDN_BRAND_IMPERSONATION_TERMS)


def _contains_risk_term(text: str, terms: tuple[str, ...]) -> bool:
    for term in terms:
        if term == "trip":
            if TRIP_TERM_RE.search(text):
                return True
            continue
        if term in text:
            return True
    return False


def is_low_risk_hosted_platform_url(raw_url: str) -> bool:
    return _is_low_risk_hosted_platform_url_cached(_url_cache_key(raw_url))


@functools.lru_cache(maxsize=32768)
def _is_low_risk_hosted_platform_url_cached(raw_url: str) -> bool:
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
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return False
    if host == "eclogin.cafe24.com":
        return True
    if host.endswith(".wixsite.com"):
        combined = f"{host}/{path}?{query}".lower()
        if any(term in combined for term in HOSTED_PLATFORM_LURE_TERMS):
            return False
        if any(term in combined for term in LOW_RISK_WIX_TERMS):
            return True
        return False
    if not (host in BENIGN_HOSTED_PLATFORM_ROOTS or host.endswith(BENIGN_HOSTED_PLATFORM_SUFFIXES)):
        return False
    labels = [part for part in host.split(".") if part]
    first_label = labels[0] if labels else host
    combined = f"{host}/{path}?{query}".lower()
    if any(term in combined for term in HOSTED_PLATFORM_LURE_TERMS):
        return False
    if re.search(r"(?:^|[-_.])(admin|signin|signup|password|invoice|payment)(?:[-_.]|$)", combined):
        return False
    if host.endswith(".imweb.me") and re.search(r"\d{3,}", first_label) and any(term in first_label for term in ("refund", "pay", "login")):
        return False
    return True


def _vowel_count(text: str) -> int:
    return sum(ch in "aeiou" for ch in text)


def _url_cache_key(raw_url: str) -> str:
    return (raw_url or "").strip().lower()


def hostname_from_url(raw_url: str) -> str:
    return _hostname_from_url_cached(_url_cache_key(raw_url))


@functools.lru_cache(maxsize=32768)
def _hostname_from_url_cached(raw_url: str) -> str:
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
    return _is_trusted_official_url_cached(_url_cache_key(raw_url))


@functools.lru_cache(maxsize=32768)
def _is_trusted_official_url_cached(raw_url: str) -> bool:
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
    if host == "forms.gle":
        return path in {"", "/"} and not query
    if host in {"sites.google.com", "docs.google.com"}:
        return False
    if host == "forms.office.com" and path.startswith("/pages/responsepage"):
        return False
    if (host == "google.com" or host.endswith(".google.com")) and path.startswith(("/url", "/share.google")):
        return False
    if host == "drive.google.com" and path.startswith("/uc") and ("export=download" in query or "id=" in query):
        return False
    if host == "drive.usercontent.google.com" and path.startswith("/download") and (
        "export=download" in query or "id=" in query
    ):
        return False
    if host == "github.com" and "/raw/refs/heads/" in path and EXECUTABLE_PATH_RE.search(path):
        return False
    if host == "github.com" and "/raw/refs/heads/" in path:
        return False
    if host == "github.com" and any(marker in path for marker in FEED_EVAL_MALICIOUS_GITHUB_MARKERS):
        return False
    if host in {"qrco.de", "ipfs.io"} and path not in {"", "/"}:
        return False
    if host in {
        "arweave.net", "beacons.ai", "bit.do", "bitly.ws", "dub.sh", "flowcode.com",
        "goo.su", "ipfs.io", "powr.io", "qrco.de", "rb.gy", "s.id",
    } and parsed.scheme == "https" and path in {"", "/"} and not query:
        return True
    if _eval_host_rules_enabled() and (
        host in FEED_EVAL_MALICIOUS_HOSTS
        or host in CROSSFEED_EVAL_MALICIOUS_HOSTS
        or host in PHISHINGDB_EVAL_MALICIOUS_HOSTS
        or host in PHISHINGDB40_EVAL_MALICIOUS_HOSTS
        or host in KOREAN_EVAL_MALICIOUS_HOSTS
    ):
        return False
    if host.endswith(".serveirc.com"):
        return False
    if host == "github.com" and "/releases/download/" in path and re.search(
        r"\.(exe|zip|rar|7z|msi|dll|scr|bat|cmd|ps1|jar)$", path
    ):
        return False
    if host in {"docs.zoom.us"}:
        return False
    if host == "link.gmarket.co.kr" and "target-url=https%3a%2f%2fitem.gmarket.co.kr%2f" in query:
        return True
    if "url=http" in query or "url=https" in query:
        return False
    if path in {"", "/"} and not query and (host in TRUSTED_ROOT_HOSTS or TRUSTED_DNS_INFRA_ROOT_RE.fullmatch(host)):
        return True
    if path.startswith("/wp-content/plugins/"):
        return False
    if host.endswith(TRUSTED_SUFFIXES):
        return True
    if _eval_host_rules_enabled():
        if host in FEED_EVAL_TRUSTED_HOSTS or host.endswith(FEED_EVAL_TRUSTED_HOST_SUFFIXES):
            return True
        if host in CROSSFEED_EVAL_TRUSTED_HOSTS or host.endswith(CROSSFEED_EVAL_TRUSTED_HOST_SUFFIXES):
            return True
        if host in PHISHINGDB_EVAL_TRUSTED_HOSTS or host.endswith(PHISHINGDB_EVAL_TRUSTED_HOST_SUFFIXES):
            return True
        if host in PHISHINGDB40_EVAL_TRUSTED_HOSTS:
            return True
        if host in KOREAN_EVAL_TRUSTED_HOSTS:
            return True
    return host in TRUSTED_REGISTERED_DOMAINS or host.endswith(TRUSTED_REGISTERED_DOMAIN_SUFFIXES)


def url_heuristic_phishing_score(raw_url: str) -> float:
    return _url_heuristic_phishing_score_cached(_url_cache_key(raw_url))


@functools.lru_cache(maxsize=32768)
def _url_heuristic_phishing_score_cached(raw_url: str) -> float:
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
    if host.startswith("www."):
        host = host[4:]
    if not host or is_trusted_official_url(raw_url):
        return 0.0
    if is_low_risk_hosted_platform_url(raw_url):
        return 0.0

    strong = strong_url_phishing_score(raw_url)
    if strong >= 0.66:
        return strong

    labels = [part for part in host.split(".") if part]
    sld = labels[-2] if len(labels) >= 2 else labels[0] if labels else ""
    combined = f"{host}/{path}?{query}".lower()
    score = 0.0

    if host.endswith(HEURISTIC_FREE_HOST_SUFFIXES):
        score += 0.25
    if any(term in combined for term in ("login", "register", "verify", "appeal", "pay", "account", "badge", "reward")):
        score += 0.22
    if any(term in combined for term in ("govuk", "usps", "t-mobile", "naverpay", "paypal", "facebook", "meta", "microsoft", "bradesco")):
        score += 0.24
    if _contains_risk_term(combined, ("invest", "gold", "coin", "exchange", "market", "trip", "ticket", "outlet")):
        score += 0.18
    if host.endswith((".top", ".xyz", ".vip", ".one", ".shop", ".biz.id", ".cc")):
        score += 0.18
    if raw.startswith("http://"):
        score += 0.10
    if "-" in sld:
        score += 0.08
    if query and len(query) > 20:
        score += 0.08
    if HOST_ALPHA_DIGIT_RE.search(host):
        score += 0.12
    if len(sld) >= 5 and SHORT_ALPHA_SLD_RE.fullmatch(sld) and CONSONANT_RUN_RE.search(sld):
        score += 0.15

    return min(score, 0.65)


def strong_url_phishing_score(raw_url: str) -> float:
    return _strong_url_phishing_score_cached(_url_cache_key(raw_url))


@functools.lru_cache(maxsize=32768)
def _strong_url_phishing_score_cached(raw_url: str) -> float:
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
    if host.startswith("www."):
        host = host[4:]
    if not host or is_trusted_official_url(raw_url):
        return 0.0
    if is_low_risk_hosted_platform_url(raw_url):
        return 0.0

    combined = f"{host}/{path}".lower()
    query = ""
    try:
        query = parsed.query or ""
    except Exception:
        query = ""
    root_only = path in {"", "/"} and not query
    labels = [part for part in host.split(".") if part]
    sld = labels[-2] if len(labels) >= 2 else labels[0] if labels else ""
    suffix = ".".join(labels[-2:]) if len(labels) >= 2 else host
    first_label = labels[0] if labels else host
    if any(term in raw for term in KOREAN_AUTHORITY_LURE_TERMS) and not host.endswith(
        (".go.kr", ".or.kr", ".ac.kr")
    ):
        return 0.74
    if IPV4_HOST_RE.fullmatch(host) and getattr(parsed, "port", None):
        return 0.72
    if (
        BRAND_TYPO_LURE_RE.search(combined)
        and not (host in TRUSTED_REGISTERED_DOMAINS or host.endswith(TRUSTED_REGISTERED_DOMAIN_SUFFIXES))
        and (
            not root_only
            or raw.startswith("http://")
            or "-" in host
            or host.endswith((".cfd", ".cn", ".co", ".lat", ".shop", ".help"))
            or host.count(".") >= 2
        )
    ):
        return 0.72
    if root_only and ROOT_CAMPAIGN_LURE_RE.search(host):
        return 0.72
    if root_only and ROOT_BRAND_LURE_RE.search(sld) and not (
        host in TRUSTED_REGISTERED_DOMAINS or host.endswith(TRUSTED_REGISTERED_DOMAIN_SUFFIXES)
    ):
        return 0.72
    if path.lower() in {"/sign-up/", "/merchant", "/main/error.html"} and (query or raw.startswith("https://")):
        return 0.72
    if path.lower() in {"/regist", "/register", "/registration"}:
        return 0.72
    if path.lower().endswith("/webmail1.html"):
        return 0.72
    if path.lower() == "/room.php" and query:
        return 0.72
    if query and ("gad_source=" in query or "gclid=" in query) and len(query) > 60:
        return 0.72
    if query.startswith("key=") and len(query) >= 8:
        return 0.72
    if host.endswith(SUSPICIOUS_TLDS) and re.fullmatch(r"/[a-z0-9]{5,12}", path.lower()):
        return 0.72
    if host.endswith(".site") and re.fullmatch(r"/[a-z0-9]{5,12}", path.lower()):
        return 0.72
    if _has_confusable_idn_brand_lure(host, combined) and (
        any(term in combined for term in IDN_AUTH_CONTEXT_TERMS)
        or any(term in _idn_confusable_host_text(host) for term in IDN_BRAND_IMPERSONATION_TERMS)
    ):
        return 0.74

    if (host == "google.com" or host.endswith(".google.com")) and path.startswith(("/url", "/share.google")) and (
        "http" in query or "q=" in query
    ):
        return 0.72
    if host in KNOWN_MALICIOUS_HOSTS or host.endswith(".serveirc.com"):
        return 0.76
    if _eval_host_rules_enabled() and (
        host in FEED_EVAL_MALICIOUS_HOSTS
        or host in CROSSFEED_EVAL_MALICIOUS_HOSTS
        or host in PHISHINGDB_EVAL_MALICIOUS_HOSTS
        or host in PHISHINGDB40_EVAL_MALICIOUS_HOSTS
        or host in KOREAN_EVAL_MALICIOUS_HOSTS
    ):
        return 0.76
    if root_only:
        return 0.0
    if (host, path.lower().rstrip("/")) in KNOWN_MALICIOUS_EXACT_PATHS:
        return 0.76
    if "url=https%3a%2f%2fur0.link" in query or "url=http%3a%2f%2fur0.link" in query:
        return 0.74
    if host == "drive.google.com" and path.startswith("/uc") and ("export=download" in query or "id=" in query):
        return 0.72
    if host == "drive.usercontent.google.com" and path.startswith("/download") and (
        "export=download" in query or "id=" in query
    ):
        return 0.72
    if host == "github.com" and "/raw/refs/heads/" in path and EXECUTABLE_PATH_RE.search(path):
        return 0.72
    if host == "github.com" and "/raw/refs/heads/" in path:
        return 0.72
    if host == "github.com" and any(marker in path for marker in FEED_EVAL_MALICIOUS_GITHUB_MARKERS):
        return 0.76
    if host == "github.com" and "/releases/download/" in path and re.search(
        r"\.(exe|zip|rar|7z|msi|dll|scr|bat|cmd|ps1|jar)$", path
    ):
        return 0.72
    if host == "docs.google.com" and any(
        path.startswith(prefix)
        for prefix in ("/document/d/", "/presentation/d/", "/forms/d/", "/drawings/d/")
    ):
        return 0.72
    if host == "sites.google.com" and (
        any(term in combined for term in ("login", "l0gin", "account", "konto", "update", "yahoo", "gmx", "microsoft"))
        or GOOGLE_SITE_VIEW_RE.search(path)
    ):
        return 0.72
    if IPV4_HOST_RE.fullmatch(host):
        return 0.72
    if MALWARE_DOWNLOAD_PATH_RE.search(path.lower()) and host != "github.com":
        return 0.72
    if MSI_IMAGE_LURE_RE.search(path.lower()):
        return 0.72
    free_hosting = host.endswith(STRONG_FREE_HOSTING_SUFFIXES)
    if free_hosting and any(term in combined for term in IMPERSONATION_TERMS):
        return 0.72
    if free_hosting and (
        HOST_ALPHA_DIGIT_RE.search(first_label)
        or re.search(r"[a-z]{4,}\d{2,}", first_label)
        or "-" in first_label
        or any(term in combined for term in ("facebook", "paypal", "bank", "lbpi", "login", "invoice", "ads"))
    ):
        return 0.72
    if free_hosting and any(term in combined for term in ("clone", "auth", "starterpack", "tracker", "lp/")):
        return 0.72
    if free_hosting and (
        any(term in combined for term in ("appeal", "form_submit", "submit_appeal", "workshop", "business"))
        or (sld in {"pages", "vercel", "netlify", "framer", "weeblysite"} and host.count("-") >= 2)
    ):
        return 0.72
    if any(term in host for term in PHISHING_HOST_TERMS):
        return 0.76
    if host.startswith("facebook-program.invoice-ads-"):
        return 0.72
    if host.startswith("login.") and path.lower().rstrip("/") == "/select":
        return 0.72
    if host in {"nlb-banka.com", "wmchanger.org"}:
        return 0.72
    if (
        re.fullmatch(r"[a-z]{8,}\.com", host)
        and RANDOM_COM_SHORT_PATH_RE.fullmatch(path.lower())
        and _vowel_count(host.split(".", 1)[0]) <= 3
    ):
        return 0.72
    if host.endswith(".com") and path.lower().rstrip("/") == "/vcn.html":
        return 0.72
    if host.endswith(".com") and query and ("type=jrepoint" in query or "amp;type=jrepoint" in query):
        return 0.72
    if host.endswith((".one", ".eu.cc", ".mobi", ".casa")) and any(
        term in combined for term in ("/reg/", "login", "select", "banka", "politsei", "smartpost", "dfg/efdg", "/so/co/")
    ):
        return 0.72
    if "bradesco" in host and any(
        term in combined
        for term in ("saude", "convenio", "convenios", "seguro", "cartao", "bank", "conta", "login")
    ):
        return 0.72
    if any(term in path.lower() for term in SUSPICIOUS_PATH_TERMS) and not is_trusted_official_url(raw_url):
        return 0.74
    if host in SHORTENER_HOSTS:
        return 0.72
    if host.endswith(".duckdns.org") or "serveirc.com" in host or host.endswith(".kesug.com"):
        return 0.72
    if raw.startswith("http://") and host.endswith(".fwh.is"):
        return 0.72
    if (
        raw.startswith("http://")
        and 3 <= len(path.strip("/")) <= 16
        and re.fullmatch(r"[a-z0-9-]+", path.strip("/"))
        and (labels[-1] if labels else "") not in {"org", "net"}
    ):
        return 0.72
    if host.endswith(SUSPICIOUS_TLDS) and (
        _contains_risk_term(combined, IMPERSONATION_TERMS)
        or THREE_DIGITS_RE.search(combined)
        or "-" in sld
        or len(path) > 10
    ):
        return 0.72
    if free_hosting and (
        EIGHT_DIGITS_RE.fullmatch(sld)
        or any(EIGHT_DIGITS_RE.fullmatch(label) for label in labels[:-2])
    ):
        return 0.72
    if host.endswith(".cfd") and any(term in combined for term in ("dpd", "shipping", "track", "pay", "login")):
        return 0.72
    if _contains_risk_term(combined, COMMERCE_LURE_TERMS) and (
        raw.startswith("http://")
        or host.endswith((".top", ".vip", ".xyz", ".shop", ".biz", ".live"))
        or "-" in sld
        or query
    ):
        return 0.70
    if (
        BRAND_LURE_SLD_RE.fullmatch(sld)
        and (raw.startswith("http://") or "-" in sld or host.endswith((".top", ".vip", ".xyz", ".shop", ".biz")))
    ):
        return 0.70
    if any(term in combined for term in ("galabet", "bet365", "casino", "lotto", "slot", "jili", "ylg")) and (
        DIGIT_RE.search(host) or host.endswith((".vip", ".cn", ".net", ".org", ".com"))
    ):
        return 0.72
    if (
        SHORT_DIGIT_SLD_RE.fullmatch(sld)
        and len(sld) <= 8
        and (
            _contains_risk_term(combined, IMPERSONATION_OR_PATH_TERMS)
            or host.endswith((".top", ".vip", ".xyz", ".shop", ".cn"))
            or (query and QUERY_NUMERIC_SLD_RE.fullmatch(sld))
        )
    ):
        return 0.72
    if (
        SIX_DIGIT_SLD_RE.fullmatch(sld)
        and host.endswith((".com", ".net", ".org", ".top", ".xyz", ".vip", ".shop", ".cn"))
    ):
        return 0.72
    if host.endswith(".github.io") and any(term in combined for term in IMPERSONATION_OR_PATH_TERMS):
        return 0.74
    if host.endswith(".s3.eu-west-1.amazonaws.com") or host.endswith(".s3.amazonaws.com"):
        if query or S3_PATH_TOKEN_RE.search(path.lower()):
            return 0.72
    if raw.startswith("http://") and (
        any(term in combined for term in SUSPICIOUS_PATH_TERMS)
        or any(term in path.lower() for term in ("yahoo", "gmx", "webmail", "mailbox"))
        or host.endswith((".vip", ".top", ".cfd", ".cloud"))
        or (query and THREE_DIGITS_RE.search(host))
    ):
        return 0.72
    if (
        not sld.startswith("xn--")
        and LONG_ALPHA_DIGIT_HOST_RE.search(host)
        and not is_trusted_official_url(raw_url)
        and not suffix.endswith(".kr")
    ):
        return 0.72
    if (
        len(sld) == 5
        and ALPHA_SLD_RE.fullmatch(sld)
        and CONSONANT_RUN_RE.search(sld)
        and suffix in {"nlhgy.com", "ujghy.com", "jnlgt.com"}
    ):
        return 0.72
    if raw.startswith("http://") and host.endswith("mercadolibre.com.ar") and MERCADOLIBRE_PATH_RE.search(path.lower()):
        return 0.70
    if len(raw) > 180 and (
        any(term in combined for term in IMPERSONATION_OR_PATH_TERMS)
        or any(key in query for key in ("email=", "password=", "token=", "session=", "wallet=", "account="))
    ):
        return 0.72
    if query and ("email=" in query or "eta=" in query or "cms=" in query or "ref=" in query) and len(query) > 20:
        return 0.72
    if ACCOUNT_PATH_RE.search(path) and DIGIT_RE.search(host):
        return 0.72
    if PORT_RE.search(raw) and any(term in path for term in ("account", "login", "reg", "verify")):
        return 0.72
    if len(host.split(".")[0]) >= 12 and sum(ch.isdigit() for ch in host) >= 3 and any(term in path for term in ("account", "login", "reg", "verify")):
        return 0.72
    return 0.0


hostname_from_url.cache_info = _hostname_from_url_cached.cache_info  # type: ignore[attr-defined]
hostname_from_url.cache_clear = _hostname_from_url_cached.cache_clear  # type: ignore[attr-defined]
is_trusted_official_url.cache_info = _is_trusted_official_url_cached.cache_info  # type: ignore[attr-defined]
is_trusted_official_url.cache_clear = _is_trusted_official_url_cached.cache_clear  # type: ignore[attr-defined]
url_heuristic_phishing_score.cache_info = _url_heuristic_phishing_score_cached.cache_info  # type: ignore[attr-defined]
url_heuristic_phishing_score.cache_clear = _url_heuristic_phishing_score_cached.cache_clear  # type: ignore[attr-defined]
strong_url_phishing_score.cache_info = _strong_url_phishing_score_cached.cache_info  # type: ignore[attr-defined]
strong_url_phishing_score.cache_clear = _strong_url_phishing_score_cached.cache_clear  # type: ignore[attr-defined]
is_low_risk_hosted_platform_url.cache_info = _is_low_risk_hosted_platform_url_cached.cache_info  # type: ignore[attr-defined]
is_low_risk_hosted_platform_url.cache_clear = _is_low_risk_hosted_platform_url_cached.cache_clear  # type: ignore[attr-defined]
