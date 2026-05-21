import json

with open('dev/test_results_local.json', encoding='utf-8') as f:
    d = json.load(f)

def print_table(rows, label):
    print('\n' + '='*118)
    print('  ' + label)
    print('='*118)
    print(f"{'No':>3}  {'XGBoost':>7}  {'GNN':>6}  {'KoBERT':>7}  {'합계':>6}  {'XG판정':<10} {'GNN판정':<10} {'KB판정':<8}  URL")
    print('-'*118)
    for i, r in enumerate(rows, 1):
        xg  = f"{r['xg']:>6.1f}" if r.get('xg')    is not None else '   N/A'
        gn  = f"{r['gn']:>5.1f}" if r.get('gn')    is not None else '  N/A'
        kb  = f"{r['kb']:>6.1f}" if r.get('kb')    is not None else '   N/A'
        tot = f"{r['total']:>5.1f}" if r.get('total') is not None else '  N/A'
        xv  = (r.get('xg_v') or 'N/A')[:10].ljust(10)
        gv  = (r.get('gn_v') or 'N/A')[:10].ljust(10)
        kv  = (r.get('kb_v') or 'N/A')[:8].ljust(8)
        url = r['url'][:55]
        print(f'{i:>3}  {xg}  {gn}  {kb}  {tot}  {xv} {gv} {kv}  {url}')
    print('-'*118)
    xgs  = [r['xg']    for r in rows if r.get('xg')    is not None]
    gns  = [r['gn']    for r in rows if r.get('gn')    is not None]
    kbs  = [r['kb']    for r in rows if r.get('kb')    is not None]
    tots = [r['total'] for r in rows if r.get('total') is not None]
    avg = lambda l: sum(l)/len(l) if l else 0
    mn  = lambda l: min(l) if l else 0
    mx  = lambda l: max(l) if l else 0
    print(f"\n  [통계]  XGBoost: avg={avg(xgs):.1f} min={mn(xgs):.1f} max={mx(xgs):.1f} ({len(xgs)}개)")
    print(f"          GNN:     avg={avg(gns):.1f} min={mn(gns):.1f} max={mx(gns):.1f} ({len(gns)}개)")
    print(f"          KoBERT:  avg={avg(kbs):.1f} min={mn(kbs):.1f} max={mx(kbs):.1f} ({len(kbs)}개)")
    print(f"          합계:    avg={avg(tots):.1f} min={mn(tots):.1f} max={mx(tots):.1f} ({len(tots)}개)")
    n = len(rows)
    xm = sum(1 for r in rows if r.get('xg_v') == 'malicious')
    gm = sum(1 for r in rows if r.get('gn_v') == 'malicious')
    km = sum(1 for r in rows if r.get('kb_v') == 'malicious')
    print(f"\n  [악성판정률]  XGBoost: {xm}/{n} ({100*xm//n}%)  GNN: {gm}/{n} ({100*gm//n}%)  KoBERT: {km}/{n} ({100*km//n}%)")

print_table(d['malicious'], '악성 사이트 50개 — XGBoost / GNN / KoBERT (각 100점 만점, 합계 300점)')
print_table(d['benign'],    '정상 사이트 50개 — XGBoost / GNN / KoBERT (각 100점 만점, 합계 300점)')
