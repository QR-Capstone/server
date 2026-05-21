"""로컬 서버(127.0.0.1:8001) 대상 배치 테스트 - XGBoost + GNN + KoBERT"""
import argparse, json, os, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "http://127.0.0.1:8001"
TIMEOUT = 60  # 로컬이라 KoBERT도 기다림
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = os.path.join(BASE, "dev")

BENIGN = [
    "https://www.naver.com","https://www.kakao.com","https://www.google.com",
    "https://www.coupang.com","https://www.gmarket.co.kr","https://www.11st.co.kr",
    "https://www.samsung.com/kr/","https://www.lge.com/kr","https://www.hyundai.com/kr/ko",
    "https://www.kia.com/kr/main.html","https://www.lotte.com","https://www.shinsegae.com",
    "https://www.apple.com/kr/","https://www.microsoft.com/ko-kr","https://www.amazon.com",
    "https://www.youtube.com","https://www.instagram.com","https://www.github.com",
    "https://www.cloudflare.com","https://www.notion.so","https://www.kakaobank.com",
    "https://www.shinhan.com","https://www.wooribank.com","https://www.kbstar.com",
    "https://www.hanabank.com","https://nhbank.co.kr","https://www.ibk.co.kr",
    "https://www.nts.go.kr","https://www.mss.go.kr","https://www.nhis.or.kr",
    "https://store.naver.com","https://pay.naver.com","https://www.kakaocorp.com",
    "https://www.sktelecom.com","https://www.kt.com","https://www.lguplus.com",
    "https://www.interpark.com","https://www.tmon.co.kr","https://www.wemakeprice.com",
    "https://www.auction.co.kr","https://www.emart.com","https://www.homeplus.co.kr",
    "https://www.lottemart.com","https://www.bgfretail.com","https://www.gs25.com",
    "https://www.cj.net","https://www.hanjin.com","https://www.korail.com",
    "https://www.airport.kr","https://www.kia.com","https://www.cgv.co.kr",
    "https://www.megabox.co.kr","https://www.lottcinema.co.kr","https://www.daum.net",
    "https://news.naver.com","https://sports.naver.com","https://map.naver.com",
    "https://finance.naver.com","https://www.toss.im","https://www.karrotpay.com",
    "https://www.payco.com","https://www.kakaopay.com","https://shopping.naver.com",
    "https://travel.naver.com","https://blog.naver.com","https://cafe.naver.com",
    "https://www.jobkorea.co.kr","https://www.saramin.co.kr","https://www.wanted.co.kr",
    "https://www.yes24.com","https://www.kyobo.co.kr","https://www.aladin.co.kr",
    "https://www.melon.com","https://www.genie.co.kr","https://www.bugs.co.kr",
    "https://www.baemin.com","https://www.yogiyo.co.kr","https://www.coupangeats.com",
    "https://www.musinsa.com","https://www.29cm.co.kr","https://www.wconcept.co.kr",
    "https://www.zigbang.com","https://www.dabangapp.com","https://www.직방.com",
    "https://www.naver.com/main.nhn","https://www.kakao.com/talk",
    "https://www.samsung.com/global/galaxy/","https://www.hyundaicard.com",
    "https://www.lottecard.co.kr","https://www.samsungcard.com",
    "https://www.kbcard.com","https://www.shinhancard.com",
    "https://www.hanwha.com","https://www.lgchem.com","https://www.posco.com",
    "https://www.sk.com","https://www.gs.co.kr","https://www.lottechilsung.co.kr",
    "https://www.ottogi.co.kr","https://www.nongshim.com","https://www.cj.co.kr",
    "https://www.emart24.co.kr","https://www.cu.co.kr","https://www.ministop.co.kr",
    "https://www.hyundaidepartment.com","https://www.lottedepartment.com",
    "https://www.gsshop.com","https://www.cjonstyle.com","https://www.lotteon.com",
]

def _call(endpoint, url):
    data = json.dumps({"url": url}).encode()
    req = urllib.request.Request(f"{API}{endpoint}", data=data,
                                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

def test_one(url):
    r = {"url": url}
    # XGBoost
    xg = _call("/analyze/xgboost", url)
    xgd = xg.get("xgboost") or {}
    p = xgd.get("final_probability")
    r["xg"] = round(float(p)*100,1) if p is not None else None
    r["xg_v"] = xgd.get("verdict","error")
    # GNN
    gn = _call("/analyze/gnn", url)
    gnd = gn.get("gnn") or {}
    p = gnd.get("probability")
    r["gn"] = round(float(p)*100,1) if p is not None else None
    r["gn_v"] = gnd.get("verdict","error")
    # KoBERT (로컬이라 시도)
    kb = _call("/analyze/engine", url)
    if "error" in kb and "judgment" not in kb:
        r["kb"] = None; r["kb_v"] = "unavail"
    else:
        rl = str(kb.get("riskLevel") or kb.get("risklevel") or "").upper()
        jg = str(kb.get("judgment") or "").lower()
        if rl=="DANGEROUS" or jg in ("unnormal","phishing","malicious"):
            r["kb"]=75.0; r["kb_v"]="malicious"
        elif rl in ("SAFE","LOW") or jg in ("normal","benign"):
            r["kb"]=10.0; r["kb_v"]="benign"
        else:
            r["kb"]=50.0; r["kb_v"]="unknown"
    scores = [s for s in [r["xg"],r["gn"],r["kb"]] if s is not None]
    r["total"] = round(sum(scores),1) if scores else None
    return r

def run(urls, label, workers=12):
    print(f"\n{'='*60}\n{label} ({len(urls)}개)\n{'='*60}")
    results = [None]*len(urls)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fm = {pool.submit(test_one, u): i for i,u in enumerate(urls)}
        done=0
        for fut in as_completed(fm):
            idx=fm[fut]; done+=1
            try: results[idx]=fut.result()
            except Exception as e: results[idx]={"url":urls[idx],"error":str(e)}
            if done%20==0 or done==len(urls):
                print(f"  [{done}/{len(urls)}]", flush=True)
    return [r for r in results if r]

def table(rows, label):
    print(f"\n{'='*110}\n  {label}\n{'='*110}")
    print(f"{'No':>3} {'XGBoost':>8} {'GNN':>7} {'KoBERT':>8} {'합계':>7}  {'XG판정':^10} {'GNN판정':^10} {'KB판정':^10}  URL")
    print("-"*110)
    for i,r in enumerate(rows,1):
        xg = f"{r['xg']:>6.1f}" if r.get('xg') is not None else "   N/A"
        gn = f"{r['gn']:>5.1f}" if r.get('gn') is not None else "  N/A"
        kb = f"{r['kb']:>6.1f}" if r.get('kb') is not None else "   N/A"
        tot= f"{r['total']:>5.1f}" if r.get('total') is not None else "  N/A"
        print(f"{i:>3} {xg} {gn} {kb} {tot}  {(r.get('xg_v','?'))[:10]:^10} {(r.get('gn_v','?'))[:10]:^10} {(r.get('kb_v','?'))[:10]:^10}  {r['url'][:55]}")
    print("-"*110)
    xgs=[r['xg'] for r in rows if r.get('xg') is not None]
    gns=[r['gn'] for r in rows if r.get('gn') is not None]
    kbs=[r['kb'] for r in rows if r.get('kb') is not None]
    tots=[r['total'] for r in rows if r.get('total') is not None]
    avg=lambda l: round(sum(l)/len(l),1) if l else 0
    mn=lambda l: round(min(l),1) if l else 0
    mx=lambda l: round(max(l),1) if l else 0
    print(f"\n  [통계]")
    print(f"  {'모델':<13}{'평균':>7}{'최소':>7}{'최대':>7}{'응답':>6}")
    print(f"  {'-'*42}")
    print(f"  {'XGBoost':<13}{avg(xgs):>7.1f}{mn(xgs):>7.1f}{mx(xgs):>7.1f}{len(xgs):>6}")
    print(f"  {'GNN':<13}{avg(gns):>7.1f}{mn(gns):>7.1f}{mx(gns):>7.1f}{len(gns):>6}")
    print(f"  {'KoBERT':<13}{avg(kbs):>7.1f}{mn(kbs):>7.1f}{mx(kbs):>7.1f}{len(kbs):>6}")
    print(f"  {'합계(300pt)':<13}{avg(tots):>7.1f}{mn(tots):>7.1f}{mx(tots):>7.1f}{len(tots):>6}")
    n=len(rows)
    xm=sum(1 for r in rows if r.get('xg_v')=='malicious')
    gm=sum(1 for r in rows if r.get('gn_v')=='malicious')
    km=sum(1 for r in rows if r.get('kb_v')=='malicious')
    print(f"\n  [악성 판정 비율] XGBoost: {xm}/{n} ({100*xm//n}%)  GNN: {gm}/{n} ({100*gm//n}%)  KoBERT: {km}/{n} ({100*km//n}%)")

def accuracy(mal_rows, ben_rows):
    specs = [("XGBoost", "xg_v"), ("GNN", "gn_v"), ("KoBERT", "kb_v")]
    print("\n[모델별 정확도]")
    for name, key in specs:
        valid = []
        for row in mal_rows:
            pred = row.get(key)
            if pred not in (None, "error", "unavail", "unknown"):
                valid.append(("malicious", pred))
        for row in ben_rows:
            pred = row.get(key)
            if pred not in (None, "error", "unavail", "unknown"):
                valid.append(("benign", pred))
        correct = sum(
            1
            for expected, pred in valid
            if (expected == "malicious" and pred == "malicious")
            or (expected == "benign" and pred == "benign")
        )
        acc = correct / len(valid) if valid else 0.0
        coverage = len(valid) / (len(mal_rows) + len(ben_rows)) if (mal_rows or ben_rows) else 0.0
        print(f"  {name:<8} accuracy={acc:.3f} valid={len(valid)} coverage={coverage:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default=API)
    parser.add_argument("--malicious-file", default=os.path.join(DEV, "mal_final.txt"))
    parser.add_argument("--out", default=os.path.join(DEV, "test_results_live.json"))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    API = args.api

    with open(args.malicious_file, encoding="utf-8") as f:
        mal_urls = [u.strip() for u in f if u.strip()]
    ben_urls = list(dict.fromkeys(BENIGN))

    print(f"악성: {len(mal_urls)}개 / 정상: {len(ben_urls)}개")
    print(f"API: {API}  (KoBERT 로컬 포함)")

    t0 = time.time()
    mal_r = run(mal_urls, "악성 사이트", workers=args.workers)
    ben_r = run(ben_urls, "정상 사이트", workers=args.workers)
    elapsed = time.time()-t0

    table(mal_r, "악성 사이트 (Malicious) 결과")
    table(ben_r, "정상 사이트 (Benign) 결과")
    accuracy(mal_r, ben_r)
    print(f"\n총 소요: {elapsed:.0f}초")

    with open(args.out,"w",encoding="utf-8") as f:
        json.dump({"malicious":mal_r,"benign":ben_r},f,ensure_ascii=False,indent=2)
    print(f"저장: {args.out}")
