#!/usr/bin/env python3
"""
URL 수집 스크립트 - PhishTank / OpenPhish / URLhaus / PhishStats / KISA / Tranco
악성 500개 + 정상 500개 수집 후 CSV 저장

출력:
  train_urls_all.csv       - XGBoost / GNN용 (전체)
  train_urls_korean.csv    - KoBERT용 (한국어 사이트만)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import re
import socket
import ssl
import time
import urllib.request
import urllib.error
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HANGUL = re.compile(r'[가-힣ᄀ-ᇿ㄰-㆏]')


# ──────────────────────────────────────────────
# 1. 피드 다운로드
# ──────────────────────────────────────────────

def _fetch(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_openphish() -> List[str]:
    """OpenPhish 무료 피드"""
    print("[openphish] 다운로드 중...")
    try:
        raw = _fetch("https://openphish.com/feed.txt", timeout=30)
        urls = [u.strip() for u in raw.decode("utf-8", errors="replace").splitlines() if u.strip().startswith("http")]
        print(f"[openphish] {len(urls)}개")
        return urls
    except Exception as e:
        print(f"[openphish] 실패: {e}")
        return []


def fetch_phishtank() -> List[str]:
    """PhishTank 공개 CSV (API 키 없이 시도)"""
    print("[phishtank] 다운로드 중...")
    try:
        raw = _fetch("https://data.phishtank.com/data/online-valid.csv.gz", timeout=60)
        with gzip.open(io.BytesIO(raw)) as gz:
            text = gz.read().decode("utf-8", errors="replace")
        urls = []
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            u = (row.get("url") or row.get("phish_url") or "").strip()
            if u.startswith("http"):
                urls.append(u)
        print(f"[phishtank] {len(urls)}개")
        return urls
    except Exception as e:
        print(f"[phishtank] 실패 (API 키 필요할 수 있음): {e}")
        return []


def fetch_urlhaus() -> List[str]:
    """URLhaus (abuse.ch) - 인증 없이 사용 가능"""
    print("[urlhaus] 다운로드 중...")
    try:
        raw = _fetch("https://urlhaus.abuse.ch/downloads/csv_recent/", timeout=30)
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                name = [n for n in zf.namelist() if n.endswith(".csv")][0]
                text = zf.read(name).decode("utf-8", errors="replace")
        except zipfile.BadZipFile:
            text = raw.decode("utf-8", errors="replace")
        urls = []
        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                u = parts[2].strip().strip('"')
                if u.startswith("http"):
                    urls.append(u)
        print(f"[urlhaus] {len(urls)}개")
        return urls
    except Exception as e:
        print(f"[urlhaus] 실패: {e}")
        return []


def fetch_phishstats_kr() -> List[str]:
    """PhishStats API - .kr 도메인 피싱 URL"""
    print("[phishstats-kr] 다운로드 중...")
    urls = []
    try:
        api = "https://phishstats.info:2096/api/phishing?_where=(url,like,%.kr%)&_size=500&_sort=-id"
        raw = _fetch(api, timeout=30)
        data = json.loads(raw.decode("utf-8", errors="replace"))
        for item in data:
            u = (item.get("url") or "").strip()
            if u.startswith("http"):
                urls.append(u)
        print(f"[phishstats-kr] {len(urls)}개")
    except Exception as e:
        print(f"[phishstats-kr] 실패: {e}")

    # 추가: 한국 IP 범위 피싱
    try:
        api2 = "https://phishstats.info:2096/api/phishing?_where=(countrycode,eq,KR)&_size=500&_sort=-id"
        raw2 = _fetch(api2, timeout=30)
        data2 = json.loads(raw2.decode("utf-8", errors="replace"))
        for item in data2:
            u = (item.get("url") or "").strip()
            if u.startswith("http"):
                urls.append(u)
        print(f"[phishstats-kr-cc] 추가 {len(data2)}개")
    except Exception as e:
        print(f"[phishstats-kr-cc] 실패: {e}")

    urls = list(dict.fromkeys(urls))
    return urls


def fetch_kisa_phishing() -> List[str]:
    """KISA (krcert.or.kr) 피싱주의보 URL 목록 스크래핑"""
    print("[kisa] 피싱주의보 스크래핑 중...")
    urls = []
    try:
        # KISA 피싱 사이트 신고 목록 페이지
        pages_to_try = [
            "https://www.krcert.or.kr/data/phishingList.do",
            "https://www.krcert.or.kr/data/maliciousList.do",
        ]
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        for page_url in pages_to_try:
            try:
                req = urllib.request.Request(page_url, headers={
                    "User-Agent": UA,
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "ko-KR,ko;q=0.9",
                })
                with urllib.request.urlopen(req, timeout=15, context=ctx) as r:
                    html = r.read().decode("utf-8", errors="replace")
                # HTML에서 http로 시작하는 URL 추출
                found = re.findall(r'https?://[^\s"\'<>]{10,}', html)
                for u in found:
                    u = u.rstrip('.,;)')
                    if u.startswith("http") and "krcert" not in u and "go.kr" not in u:
                        urls.append(u)
            except Exception as e:
                print(f"[kisa] {page_url} 실패: {e}")

    except Exception as e:
        print(f"[kisa] 실패: {e}")

    urls = list(dict.fromkeys(urls))
    print(f"[kisa] {len(urls)}개")
    return urls


def fetch_nurilab() -> List[str]:
    """누리랩(Nurilab) 피싱주의보 - 한국어 피싱 사이트 목록"""
    print("[nurilab] 피싱주의보 스크래핑 중...")
    urls = []
    dot_pattern = re.compile(r'(\[\.\]|\(dot\)|\s+\.\s+)', re.I)  # [.] → . 역변환
    hxxp_pattern = re.compile(r'hxxps?://[^\s"\'<>]+', re.I)
    url_pattern = re.compile(r'https?://[^\s"\'<>]+', re.I)
    years = ["2026", "2025", "2024", "2023", "2022"]
    base = "https://www.nurilab.com/kr/alert/phishing_alert/phishing_alert_paper/"

    for year in years:
        list_url = f"{base}phishing_alert_list_{year}.html"
        try:
            raw = _fetch(list_url, timeout=30)
            html = raw.decode("utf-8", errors="replace")
            found = set()
            found.update(re.findall(r'title="([^"]{5,})"', html))
            found.update(url_pattern.findall(html))
            found.update(hxxp_pattern.findall(html))
            added = 0
            for raw_u in found:
                u = dot_pattern.sub('.', raw_u).strip().rstrip('.,;)')
                u = re.sub(r'^hxxp', 'http', u, flags=re.I)
                if u.startswith("http") and "nurilab.com" not in u and "tistory.com" not in u:
                    urls.append(u)
                    added += 1
            print(f"[nurilab] {year}: {added}개")
        except Exception as e:
            print(f"[nurilab] {year} 실패: {e}")

    # 누리랩 공식 블로그/보도자료에도 월간 피싱주의보가 게시되어 보조 수집원으로 사용한다.
    for blog_url in [
        "https://nurilab.tistory.com/category/%ED%94%BC%EC%8B%B1%EC%A3%BC%EC%9D%98%EB%B3%B4",
        "https://nurilab.tistory.com/category/AskURL",
    ]:
        try:
            html = _fetch(blog_url, timeout=20).decode("utf-8", errors="replace")
            added = 0
            for raw_u in set(url_pattern.findall(html)) | set(hxxp_pattern.findall(html)):
                u = dot_pattern.sub('.', raw_u).strip().rstrip('.,;)')
                u = re.sub(r'^hxxp', 'http', u, flags=re.I)
                if u.startswith("http") and "nurilab" not in u and "tistory" not in u:
                    urls.append(u)
                    added += 1
            print(f"[nurilab-blog] {added}개")
        except Exception as e:
            print(f"[nurilab-blog] 실패: {e}")

    urls = list(dict.fromkeys(urls))
    print(f"[nurilab] 총 {len(urls)}개")
    return urls


def fetch_phishing_database_kr() -> List[str]:
    """GitHub 오픈소스 피싱 DB - 한국 관련"""
    print("[phishing-db] 다운로드 중...")
    urls = []
    sources = [
        # Mitchell Krogza's phishing database
        "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt",
    ]
    for src in sources:
        try:
            raw = _fetch(src, timeout=30)
            lines = raw.decode("utf-8", errors="replace").splitlines()
            for line in lines:
                line = line.strip()
                if line.startswith("http"):
                    urls.append(line)
                elif re.match(r'^[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}$', line):
                    urls.append(f"http://{line}")
        except Exception as e:
            print(f"[phishing-db] {src} 실패: {e}")

    urls = list(dict.fromkeys(urls))
    print(f"[phishing-db] {len(urls)}개")
    return urls


def fetch_tranco(n: int = 1000) -> List[str]:
    """Tranco Top 1M에서 정상 사이트"""
    print("[tranco] 다운로드 중...")
    try:
        raw = _fetch("https://tranco-list.eu/top-1m.csv.zip", timeout=60)
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            name = zf.namelist()[0]
            text = zf.read(name).decode("utf-8", errors="replace")
        urls = []
        for line in text.splitlines()[:n * 3]:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                domain = parts[1].strip()
                if domain and "." in domain:
                    urls.append(f"https://{domain}")
        print(f"[tranco] {len(urls)}개 후보")
        return urls[:n * 2]
    except Exception as e:
        print(f"[tranco] 실패: {e}")
        return []


# ──────────────────────────────────────────────
# 2. 생존 여부 + 한국어 여부 확인
# ──────────────────────────────────────────────

def check_url(url: str, timeout: int = 8) -> Tuple[bool, bool]:
    """
    Returns (is_alive, is_korean)
    is_korean: .kr TLD 또는 응답에 한글 포함
    """
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(
            url,
            headers={"User-Agent": UA, "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8"},
        )
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            if resp.status >= 400:
                return False, False
            chunk = resp.read(4096).decode("utf-8", errors="replace")
            is_kr_tld = bool(re.search(r'\.kr[/"\':\s]|\.kr$|\.kr/', url, re.I))
            has_hangul = bool(HANGUL.search(chunk))
            # 한국어로 판단하는 추가 조건: 한국 호스팅/도메인
            is_kr_domain = bool(re.search(
                r'\.(co\.kr|or\.kr|go\.kr|ne\.kr|re\.kr|pe\.kr|mil\.kr|ac\.kr|hs\.kr|ms\.kr|es\.kr|sc\.kr|kg\.kr|seoul\.kr|busan\.kr|daegu\.kr|incheon\.kr|gwangju\.kr|daejeon\.kr|ulsan\.kr|sejong\.kr|gyeonggi\.kr|gangwon\.kr|chungbuk\.kr|chungnam\.kr|jeonbuk\.kr|jeonnam\.kr|gyeongbuk\.kr|gyeongnam\.kr|jeju\.kr)([/"\':\s]|$)',
                url, re.I
            ))
            return True, (is_kr_tld or has_hangul or is_kr_domain)
    except Exception:
        return False, False


def filter_alive(urls: List[str], label: int, workers: int = 20, max_count: int = 500) -> List[dict]:
    """병렬로 생존 확인 — 배치 단위로 처리해 목표 달성 즉시 종료"""
    results: List[dict] = []
    total = len(urls)
    batch_size = workers * 8  # 배치당 최대 future 수

    for batch_start in range(0, total, batch_size):
        if len(results) >= max_count:
            break
        batch = urls[batch_start: batch_start + batch_size]
        checked_base = batch_start

        with ThreadPoolExecutor(max_workers=workers) as pool:
            fmap = {pool.submit(check_url, url): url for url in batch}
            for fut in as_completed(fmap):
                url = fmap[fut]
                checked_base += 1
                try:
                    alive, is_korean = fut.result()
                    if alive:
                        results.append({"url": url, "label": label, "is_korean": is_korean})
                        if len(results) % 20 == 0:
                            print(f"  살아있는 사이트: {len(results)}개 (확인: ~{checked_base}개 / 전체: {total}개)")
                        if len(results) >= max_count:
                            return results
                except Exception:
                    pass

    return results


# ──────────────────────────────────────────────
# 3. CSV 저장
# ──────────────────────────────────────────────

def save_csv(rows: List[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label", "source", "is_korean"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "url": row["url"],
                "label": row["label"],
                "source": row.get("source", "auto"),
                "is_korean": "1" if row.get("is_korean") else "0",
            })
    print(f"저장: {path} ({len(rows)}행)")


# ──────────────────────────────────────────────
# 4. main
# ──────────────────────────────────────────────

# 한국 주요 정상 사이트 (KoBERT 학습용 확장 목록)
KR_BENIGN_SITES = [
    # 포털/검색
    "https://www.naver.com", "https://www.kakao.com", "https://www.daum.net",
    "https://www.nate.com", "https://news.naver.com", "https://sports.naver.com",
    "https://map.naver.com", "https://finance.naver.com", "https://blog.naver.com",
    "https://cafe.naver.com", "https://shopping.naver.com", "https://travel.naver.com",
    # 쇼핑
    "https://www.coupang.com", "https://www.gmarket.co.kr", "https://www.11st.co.kr",
    "https://www.auction.co.kr", "https://www.interpark.com", "https://www.tmon.co.kr",
    "https://www.wemakeprice.com", "https://www.lotteon.com", "https://www.gsshop.com",
    "https://www.cjonstyle.com", "https://www.hmall.com", "https://www.ssg.com",
    "https://www.musinsa.com", "https://www.29cm.co.kr", "https://www.wconcept.co.kr",
    "https://www.yes24.com", "https://www.kyobo.co.kr", "https://www.aladin.co.kr",
    # 은행
    "https://www.shinhan.com", "https://www.kbstar.com", "https://www.wooribank.com",
    "https://www.hanabank.com", "https://nhbank.co.kr", "https://www.ibk.co.kr",
    "https://www.kakaobank.com", "https://www.tossbank.com", "https://www.kbanknow.com",
    "https://www.sc.com/kr", "https://www.citibank.co.kr",
    # 카드
    "https://www.shinhancard.com", "https://www.kbcard.com", "https://www.hyundaicard.com",
    "https://www.lottecard.co.kr", "https://www.samsungcard.com", "https://card.wooribank.com",
    "https://www.hanacard.co.kr", "https://www.nonghyupcard.com", "https://www.ibkcard.com",
    # 통신
    "https://www.sktelecom.com", "https://www.kt.com", "https://www.lguplus.com",
    "https://www.sktelink.com", "https://www.ktm.kr",
    # 정부/공공
    "https://www.nts.go.kr", "https://www.mss.go.kr", "https://www.nhis.or.kr",
    "https://www.hometax.go.kr", "https://www.gov.kr", "https://www.moe.go.kr",
    "https://www.moef.go.kr", "https://www.mois.go.kr", "https://www.mohw.go.kr",
    "https://www.police.go.kr", "https://www.nia.or.kr", "https://www.kisa.or.kr",
    "https://www.krcert.or.kr", "https://www.npa.go.kr", "https://www.customs.go.kr",
    # 교통/여행
    "https://www.korail.com", "https://www.airport.kr", "https://www.subway.co.kr",
    "https://www.letskorail.com", "https://www.srt.co.kr",
    # 제조/기업
    "https://www.samsung.com/kr/", "https://www.lge.com/kr",
    "https://www.hyundai.com/kr/ko", "https://www.kia.com/kr/",
    "https://www.sk.com", "https://www.gs.co.kr", "https://www.lotte.com",
    "https://www.shinsegae.com", "https://www.cj.net", "https://www.hanwha.com",
    "https://www.posco.com", "https://www.lgchem.com",
    # 편의점/유통
    "https://www.gs25.com", "https://www.bgfretail.com", "https://www.cu.co.kr",
    "https://www.ministop.co.kr", "https://www.emart24.co.kr",
    "https://www.emart.com", "https://www.homeplus.co.kr", "https://www.lottemart.com",
    "https://www.hyundaidepartment.com", "https://www.lottedepartment.com",
    # 음식/배달
    "https://www.baemin.com", "https://www.yogiyo.co.kr", "https://www.coupangeats.com",
    # 영화/엔터
    "https://www.cgv.co.kr", "https://www.megabox.co.kr", "https://www.lottcinema.co.kr",
    "https://www.melon.com", "https://www.genie.co.kr", "https://www.bugs.co.kr",
    "https://www.flo.io",
    # 구직
    "https://www.jobkorea.co.kr", "https://www.saramin.co.kr", "https://www.wanted.co.kr",
    "https://www.incruit.com", "https://www.rallit.com",
    # 부동산
    "https://www.zigbang.com", "https://www.dabangapp.com",
    "https://www.peterpanz.com", "https://www.naver.com/realestate",
    # 핀테크/페이
    "https://www.toss.im", "https://www.kakaopay.com", "https://pay.naver.com",
    "https://www.payco.com", "https://www.karrotpay.com",
    # 언론
    "https://www.chosun.com", "https://www.donga.com", "https://www.joongang.co.kr",
    "https://www.hani.co.kr", "https://www.khan.co.kr", "https://www.ytn.co.kr",
    "https://www.mbc.co.kr", "https://www.kbs.co.kr", "https://www.sbs.co.kr",
    # 대학
    "https://www.snu.ac.kr", "https://www.yonsei.ac.kr", "https://www.korea.ac.kr",
    "https://www.kaist.ac.kr", "https://www.postech.ac.kr", "https://www.skku.edu",
    # 기타
    "https://store.naver.com", "https://www.kakaocorp.com",
    "https://www.hanjin.com", "https://www.cjlogistics.com",
    "https://www.lottechilsung.co.kr", "https://www.ottogi.co.kr",
    "https://www.nongshim.com", "https://www.cj.co.kr",
]

# 유명 대형 사이트만으로 정상셋이 치우치지 않도록 중소/지역/공공/생활 서비스도 포함한다.
KR_BENIGN_SITES += [
    "https://www.seoul.go.kr", "https://www.busan.go.kr", "https://www.daegu.go.kr",
    "https://www.incheon.go.kr", "https://www.gwangju.go.kr", "https://www.daejeon.go.kr",
    "https://www.ulsan.go.kr", "https://www.sejong.go.kr", "https://www.gg.go.kr",
    "https://www.provin.gangwon.kr", "https://www.chungbuk.go.kr", "https://www.chungnam.go.kr",
    "https://www.jeonbuk.go.kr", "https://www.jeonnam.go.kr", "https://www.gb.go.kr",
    "https://www.gyeongnam.go.kr", "https://www.jeju.go.kr",
    "https://www.kosha.or.kr", "https://www.keis.or.kr", "https://www.keco.or.kr",
    "https://www.kepco.co.kr", "https://www.kwater.or.kr", "https://www.ex.co.kr",
    "https://www.khug.or.kr", "https://www.hf.go.kr", "https://www.kibo.or.kr",
    "https://www.kotra.or.kr", "https://www.kbiz.or.kr", "https://www.korcham.net",
    "https://www.k-startup.go.kr", "https://www.sbiz.or.kr", "https://www.work24.go.kr",
    "https://www.hrd.go.kr", "https://www.epeople.go.kr", "https://www.safekorea.go.kr",
    "https://www.weather.go.kr", "https://www.data.go.kr", "https://www.law.go.kr",
    "https://www.court.go.kr", "https://www.iros.go.kr", "https://www.driver.go.kr",
    "https://www.bok.or.kr", "https://www.fss.or.kr", "https://www.kofia.or.kr",
    "https://www.krx.co.kr", "https://www.kdic.or.kr", "https://www.credit4u.or.kr",
    "https://www.epost.go.kr", "https://www.ilogen.com", "https://www.lotteglogis.com",
    "https://www.cuonet.com", "https://www.oliveyoung.co.kr", "https://www.elandmall.co.kr",
    "https://www.ohou.se", "https://www.idus.com", "https://www.brandi.co.kr",
    "https://www.styleshare.kr", "https://www.todayhouse.co.kr", "https://www.daangn.com",
    "https://www.yeogi.com", "https://www.goodchoice.kr", "https://www.yanolja.com",
    "https://www.modetour.com", "https://www.hanatour.com", "https://www.twayair.com",
    "https://www.jejuair.net", "https://www.jinair.com", "https://flyairseoul.com",
    "https://www.airbusan.com", "https://www.koreanair.com", "https://flyasiana.com",
    "https://www.hufs.ac.kr", "https://www.hanyang.ac.kr", "https://www.cau.ac.kr",
    "https://www.khu.ac.kr", "https://www.ewha.ac.kr", "https://www.pusan.ac.kr",
    "https://www.knu.ac.kr", "https://www.jbnu.ac.kr", "https://www.jnu.ac.kr",
    "https://www.yna.co.kr", "https://www.newsis.com", "https://www.fnnews.com",
    "https://www.mk.co.kr", "https://www.hankyung.com", "https://www.etnews.com",
    "https://www.boannews.com", "https://www.dailysecu.com",
]


def load_existing_urls(path: str) -> set:
    """기존 CSV에서 URL 집합 로드 (중복 제외용)"""
    existing = set()
    if not os.path.isfile(path):
        return existing
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                u = (row.get("url") or "").strip()
                if u:
                    existing.add(u)
        print(f"[exclude] 기존 URL {len(existing)}개 로드: {path}")
    except Exception as e:
        print(f"[exclude] 로드 실패: {e}")
    return existing


def merge_csvs(paths: List[str], out_path: str) -> int:
    """여러 CSV를 중복 없이 병합"""
    seen = set()
    rows = []
    for p in paths:
        if not os.path.isfile(p):
            continue
        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                u = (row.get("url") or "").strip()
                if u and u not in seen:
                    seen.add(u)
                    rows.append(row)
    if not rows:
        return 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="URL 수집 스크립트 (피싱주의보 + PhishTank)")
    parser.add_argument("--malicious", type=int, default=500, help="악성 URL 목표 개수")
    parser.add_argument("--benign", type=int, default=500, help="정상 URL 목표 개수")
    parser.add_argument("--workers", type=int, default=25)
    parser.add_argument("--out-dir", default=BASE_DIR)
    parser.add_argument("--skip-phishtank", action="store_true")
    parser.add_argument("--exclude-csv", default="", help="이미 수집된 URL CSV (중복 제외)")
    parser.add_argument("--suffix", default="", help="출력 파일명 접미사 (예: _2 → train_urls_all_2.csv)")
    args = parser.parse_args()

    # 제외 URL 로드
    exclude_urls: set = set()
    if args.exclude_csv:
        exclude_urls = load_existing_urls(args.exclude_csv)

    sfx = args.suffix  # 파일명 접미사

    # ── 악성 URL 수집 ──
    print("\n=== 악성 URL 수집 ===")
    phishing_raw: List[str] = []

    # 1. 누리랩 피싱주의보 (한국 특화, 최우선)
    phishing_raw += fetch_nurilab()

    # 2. KISA 피싱주의보
    phishing_raw += fetch_kisa_phishing()

    # 3. PhishStats .kr 필터 (한국 특화)
    phishing_raw += fetch_phishstats_kr()

    # 4. OpenPhish
    phishing_raw += fetch_openphish()

    # 4. PhishTank
    if not args.skip_phishtank:
        phishing_raw += fetch_phishtank()

    # 5. URLhaus
    if len(phishing_raw) < args.malicious * 3:
        phishing_raw += fetch_urlhaus()

    # 6. GitHub 피싱 DB
    if len(phishing_raw) < args.malicious * 2:
        phishing_raw += fetch_phishing_database_kr()

    # 중복 제거 + 기존 URL 제외
    phishing_raw = list(dict.fromkeys(phishing_raw))
    if exclude_urls:
        before = len(phishing_raw)
        phishing_raw = [u for u in phishing_raw if u not in exclude_urls]
        print(f"[exclude] 악성 후보 {before}개 → {len(phishing_raw)}개 (제외: {before - len(phishing_raw)}개)")
    print(f"\n총 피싱 후보: {len(phishing_raw)}개 → 생존 확인 시작")

    print(f"생존 확인 중 (목표: {args.malicious}개, workers={args.workers})...")
    t0 = time.time()
    mal_rows = filter_alive(phishing_raw, label=1, workers=args.workers, max_count=args.malicious)
    print(f"악성 수집 완료: {len(mal_rows)}개 (한국어: {sum(1 for r in mal_rows if r['is_korean'])}개) — {time.time()-t0:.0f}초")

    # ── 정상 URL 수집 ──
    print("\n=== 정상 URL 수집 ===")
    benign_raw: List[str] = list(KR_BENIGN_SITES)

    if args.benign > len(KR_BENIGN_SITES):
        tranco = fetch_tranco(n=args.benign * 3)
        benign_raw = benign_raw + tranco

    benign_raw = list(dict.fromkeys(benign_raw))
    if exclude_urls:
        before = len(benign_raw)
        benign_raw = [u for u in benign_raw if u not in exclude_urls]
        print(f"[exclude] 정상 후보 {before}개 → {len(benign_raw)}개 (제외: {before - len(benign_raw)}개)")
    if len(benign_raw) < args.benign:
        need = args.benign - len(benign_raw)
        print(f"[benign] 제외 후 정상 후보 부족: {len(benign_raw)}/{args.benign} → Tranco 후보 추가")
        tranco = fetch_tranco(n=max(args.benign * 3, need * 8))
        before = len(benign_raw)
        benign_raw = list(dict.fromkeys(benign_raw + [u for u in tranco if u not in exclude_urls]))
        print(f"[benign] Tranco 추가 후 정상 후보 {before}개 → {len(benign_raw)}개")
    print(f"\n정상 후보: {len(benign_raw)}개 → 생존 확인 시작")

    print(f"생존 확인 중 (목표: {args.benign}개, workers={args.workers})...")
    t0 = time.time()
    ben_rows = filter_alive(benign_raw, label=0, workers=args.workers, max_count=args.benign)
    for r in ben_rows:
        r["source"] = "kr_benign"
    print(f"정상 수집 완료: {len(ben_rows)}개 (한국어: {sum(1 for r in ben_rows if r['is_korean'])}개) — {time.time()-t0:.0f}초")

    # ── 저장 ──
    all_rows = mal_rows + ben_rows
    all_path = os.path.join(args.out_dir, f"train_urls_all{sfx}.csv")
    save_csv(all_rows, all_path)

    # KoBERT용: 한국어 사이트만
    kr_rows = [r for r in all_rows if r.get("is_korean")]
    kr_path = os.path.join(args.out_dir, f"train_urls_korean{sfx}.csv")
    save_csv(kr_rows, kr_path)

    print(f"\n=== 수집 완료 ===")
    print(f"전체:   {len(all_rows)}개 (악성 {len(mal_rows)} / 정상 {len(ben_rows)})")
    kr_mal = sum(1 for r in mal_rows if r.get("is_korean"))
    kr_ben = sum(1 for r in ben_rows if r.get("is_korean"))
    print(f"한국어: {len(kr_rows)}개 (악성 {kr_mal} / 정상 {kr_ben}) → KoBERT 학습")
    print(f"\n다음 단계:")
    print(f"  XGBoost 학습:   python xgboost/XG_train.py --input {all_path}")
    print(f"  GNN 데이터수집: python gnn/collect_gnn_dataset.py --input {all_path}")
    print(f"  GNN 학습:       python gnn/regenerate_gnn_model.py")
    print(f"  KoBERT 학습:   python KoBERT/kobert_train.py --input {kr_path}")


if __name__ == "__main__":
    main()
