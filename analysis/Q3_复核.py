# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。AI工具名称：DeepSeek；版本/型号：V4.1-Flash；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；版本发布日期：2026年9月10日。

"""
Q3 结果一致性复核程序（Q3_复核.py）
从Q2主方案架次表、逐箱交付表与原始通信/中继参数出发，独立重算：
  1) 全程轨迹（分阶段+交接段采样，每航段20点）
  2) 逐点G01双向链路判定（FSPL=32.45、Lobs=10、M=8dB、门限-90dBm、LoS 100m采样）
  3) 需中继时刻时间簇聚类（gap>400s）
  4) 受限候选悬停点集（0.005°网格、离地300m档<=上限、距O01<12km、可达性过滤）覆盖
  5) 3架中继机时序指派（最晚开始式）与能源组件周转（E<=2.56kWh、SOC>=20%、两阶段充电1800s）
输出：任务通信保障率、需中继覆盖、中继能耗、逐架次保障——与Q3_TD-CJO.py主结果比对。
口径与主程序一致（同一组题目规则与参数）；数据读取与计算路径独立实现。
"""
import os, json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from math import radians, sin, cos, sqrt, atan2, log10

import os
BASE = os.path.dirname(os.path.abspath(__file__))
DEM_MAT = BASE + r'\数据\镇龙乡地理空间数据\镇龙乡及周边地理数据\数字高程模型数据（DEM）\镇龙乡及周边30米DEM.mat'
G = 9.81
m = loadmat(DEM_MAT)
dem = np.array(m['dem'], dtype=float); lons = np.array(m['longitude']).ravel(); lats = np.array(m['latitude']).ravel()
dem[dem < 0] = np.nan
lon0, dlon = lons[0], lons[1] - lons[0]; lat0, dlat = lats[0], lats[1] - lats[0]

def dem_at(lon, lat):
    j = np.clip(((np.asarray(lon) - lon0) / dlon).astype(int), 0, len(lons) - 1)
    i = np.clip(((np.asarray(lat) - lat0) / dlat).astype(int), 0, len(lats) - 1)
    return dem[i, j]

def hav(lo1, la1, lo2, la2):
    r = lambda x: radians(x)
    a = sin((r(la2) - r(la1)) / 2) ** 2 + cos(r(la1)) * cos(r(la2)) * sin((r(lo2) - r(lo1)) / 2) ** 2
    return 6371000 * 2 * atan2(sqrt(a), sqrt(1 - a))

F = 2400.0; LSYS = 3.0; LOBS = 10.0; PSENS = -98.0; M = 8.0
PTHR = PSENS + M
PAR = {'G01': (27.0, 12.0), 'TR': (20.0, 3.0), 'REL_A': (20.0, 6.0), 'REL_B': (19.0, 8.0)}
RV = 15.0; RVUP = 4.0; RVDN = 3.0; RPCR = 1.15; RPH = 1.05; RPC = 0.05
RM = 23.5; RETA = 0.72; RBAT = 3.2; RSOC = 0.20; RTCHG = 1800.0
RPREP = 180.0; RLINK = 30.0; RTURN = 300.0

def fspl(dm): return 32.45 + 20.0 * np.log10(F) + 20.0 * np.log10(dm / 1000.0)

def blocked(a, b):
    d = hav(a[0], a[1], b[0], b[1])
    if d < 1e-9: return False
    ns = max(2, min(60, int(d / 100.0) + 1))
    ts = np.linspace(0.0, 1.0, ns + 1)[1:-1]
    return bool(np.any(np.isfinite(dem_at(a[0] + ts * (b[0] - a[0]), a[1] + ts * (b[1] - a[1])))
                        & (dem_at(a[0] + ts * (b[0] - a[0]), a[1] + ts * (b[1] - a[1])) > a[2] + ts * (b[2] - a[2]))))

def link(a, ka, b, kb):
    d3 = sqrt(hav(a[0], a[1], b[0], b[1]) ** 2 + (a[2] - b[2]) ** 2)
    pl = fspl(d3) + (LOBS if blocked(a, b) else 0.0)
    r1 = PAR[ka][0] + PAR[ka][1] + PAR[kb][1] - LSYS - pl
    return (r1 >= PTHR, r1)

def bilink(a, ka, b, kb):
    ok1, _ = link(a, ka, b, kb); ok2, _ = link(b, kb, a, ka)
    return ok1 and ok2

# ---- 输入 ----
nodes = pd.read_csv(BASE + r'\预处理输出\Q1_节点_预处理.csv')
q2 = pd.read_csv(BASE + r'\Q2_主方案20架次_架次表.csv')
cargo = pd.read_csv(BASE + r'\预处理输出\Q1_货箱明细_预处理.csv')
uav = pd.read_csv(BASE + r'\预处理输出\Q2_机型参数_预处理.csv')
dfd = pd.read_csv(BASE + r'\Q2_主方案20架次_逐箱交付.csv')
o = nodes[nodes['编号'] == 'O01'].iloc[0]
OLON, OLAT, OALT = float(o['经度']), float(o['纬度']), float(o['海拔(m)'])
sa = nodes[nodes['编号'] != 'O01'].copy().reset_index(drop=True)
G01 = (OLON, OLAT, OALT + 20.0)
pps = {u['机型编号']: dict(v=float(u['巡航速度(m/s)']), vu=float(u['最大爬升速度(m/s)']),
                           vd=float(u['最大下降速度(m/s)']), h0=float(u['基础交接时间(s)']),
                           h1=float(u['每箱增加交接时间(s)'])) for _, u in uav.iterrows()}
hop = {r['编号']: float(r['海拔(m)']) + 30.0 for _, r in sa.iterrows()}; hop['O01'] = OALT
lon = {r['编号']: float(r['经度']) for _, r in sa.iterrows()}; lat = {r['编号']: float(r['纬度']) for _, r in sa.iterrows()}
lon['O01'], lat['O01'] = OLON, OLAT
bw = {r['货箱序号']: float(r['单箱质量(kg)']) for _, r in cargo.iterrows()}

def cruise(a, b):
    ts = [0.0, 1.0]
    for ax, x1, x2 in [(lons, a[0], b[0]), (lats, a[1], b[1])]:
        bd = (ax[:-1] + ax[1:]) / 2.0
        t = (bd - x1) / (x2 - x1)
        ts += t[(t > 0) & (t < 1)].tolist()
    ts = sorted(set(ts))
    sp = [0.0, 1.0] + [(u + v) / 2.0 for u, v in zip(ts, ts[1:])]
    return max(dem_at(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])) for t in sp) + 50.0

import ast
def parse_deliv(s):
    if isinstance(s, str) and s.strip().startswith('{'):
        return {k.strip("'"): float(v) for k, v in ast.literal_eval(s.replace('np.float64(', '').replace(')', '')).items()}
    return {}

traj = []
for _, row in q2.iterrows():
    seq = row['路径'].split('→'); mt = row['机型']; pp = pps[mt]
    bz = {}
    for _, c in dfd.iterrows():
        if c['架次ID'] == row['架次ID']:
            bz.setdefault(c['服务区'], []).append(c['货箱序号'])
    t = float(row['起飞s']); rem = float(row['载重kg']); prev = 'O01'
    dlv = parse_deliv(row['交付完成']); ret = float(row['返航s'])
    for z in seq[1:-1] + ['O01']:
        d = hav(lon[prev], lat[prev], lon[z], lat[z])
        Hc = cruise((lon[prev], lat[prev]), (lon[z], lat[z]))
        h0, h1 = hop[prev], hop[z]
        tcl = max(Hc - h0, 0.0) / pp['vu']; tcr = d / pp['v']; tdn = max(Hc - h1, 0.0) / pp['vd']
        hand = 0.0 if z == 'O01' else (pp['h0'] + pp['h1'] * len(bz.get(z, [])))
        tend = ret if z == 'O01' else dlv[z] - hand
        sc = max(tend - t, 1e-9) / max(tcl + tcr + tdn, 1e-9)
        tcl, tcr, tdn = tcl * sc, tcr * sc, tdn * sc
        seg = tcl + tcr + tdn
        for k in range(20 + 1):
            tt = seg * k / 20
            if tt <= tcl:
                hg = h0 + (Hc - h0) * (tt / max(tcl, 1e-9)); lo_, la_ = lon[prev], lat[prev]
            elif tt <= tcl + tcr:
                hg = Hc; uu = (tt - tcl) / max(tcr, 1e-9)
                lo_ = lon[prev] + (lon[z] - lon[prev]) * uu; la_ = lat[prev] + (lat[z] - lat[prev]) * uu
            else:
                hg = Hc + (h1 - Hc) * ((tt - tcl - tcr) / max(tdn, 1e-9)); lo_, la_ = lon[z], lat[z]
            traj.append((t + tt, lo_, la_, hg, row['架次ID']))
        if z != 'O01':
            nh = max(2, int(hand / 60.0) + 1)
            for k in range(nh + 1):
                traj.append((tend + hand * k / nh, lon[z], lat[z], hop[z], row['架次ID']))
        t = tend + hand; prev = z
df = pd.DataFrame(traj, columns=['t', 'lon', 'lat', 'h', '架次']).drop_duplicates(subset=['t', '架次']).sort_values('t').reset_index(drop=True)
df['直连'] = [bilink(G01, 'G01', (r['lon'], r['lat'], r['h']), 'TR') for _, r in df.iterrows()]
need = df[~df['直连']].copy().reset_index(drop=True)
print(f"[复核] 总采样点 {len(df)}，直连 {int(df['直连'].sum())} ({df['直连'].mean()*100:.1f}%)，需中继 {len(need)}")

# ---- 候选 ----
cands = []
lo = float(lons.min())
while lo <= float(lons.max()):
    la = float(lats.min())
    while la <= float(lats.max()):
        e = dem_at(lo, la)
        if np.isfinite(e):
            cands.append((f"G{lo:.4f}_{la:.4f}", lo, la, e + 300.0))
        la += 0.005
    lo += 0.005
cands = [c for c in cands if hav(OLON, OLAT, c[1], c[2]) < 12000]

def rfl(c):
    d = hav(OLON, OLAT, c[1], c[2])
    Hc = cruise((OLON, OLAT), (c[1], c[2]))
    H = max(Hc, OALT, c[3])
    tg = max(H - OALT, 0.0) / RVUP + d / RV + max(H - c[3], 0.0) / RVDN
    tb = max(H - c[3], 0.0) / RVUP + d / RV + max(H - OALT, 0.0) / RVDN
    Eg = RPCR * (d / RV) / 3600.0 + G * RM * max(H - OALT, 0.0) / (RETA * 3600000)
    Eb = RPCR * (d / RV) / 3600.0 + G * RM * max(H - c[3], 0.0) / (RETA * 3600000)
    return dict(tg=tg, tb=tb, Eg=Eg, Eb=Eb, H=H)

def covers(c, p):
    pc = (c[1], c[2], c[3]); pt = (p['lon'], p['lat'], p['h'])
    return bilink(pt, 'TR', pc, 'REL_A') and bilink(pc, 'REL_B', G01, 'G01')

# ---- 聚类 ----
clu = []; cur = [need.index[0]]
for i in range(1, len(need)):
    if need.iloc[i]['t'] - need.iloc[i - 1]['t'] > 400.0:
        clu.append(cur); cur = [need.index[i]]
    else:
        cur.append(need.index[i])
clu.append(cur)
print(f"[复核] 时间簇 {len(clu)} 个:", [(round(need.loc[g]['t'].min()), round(need.loc[g]['t'].max()), len(g)) for g in clu])

# ---- 覆盖（每簇递归多段） ----
tasks = []
for ci, gi in enumerate(clu):
    c = need.loc[gi]
    t0 = float(c['t'].min()); mlon = float(c['lon'].mean()); mlat = float(c['lat'].mean())
    pool = [cd for cd in cands
            if hav(mlon, mlat, cd[1], cd[2]) < 9000
            and rfl(cd)['tg'] <= t0 - (RPREP + RLINK)]
    idxs = list(c.index); pts = [c.loc[i] for i in idxs]
    cm = np.array([[covers(cd, p) for p in pts] for cd in pool], dtype=bool)
    rem = np.ones(len(idxs), dtype=bool); seg = 0
    while rem.any():
        cnt = cm[:, rem].sum(axis=1)
        if cnt.max() <= 0: break
        best = min(np.flatnonzero(cnt == cnt.max()), key=lambda i: (rfl(pool[i])['tb'], pool[i][3]))
        cov = rem & cm[best]
        seg += 1
        cidxs = [int(idxs[i]) for i in np.flatnonzero(cov)]
        tasks.append(dict(clu=ci + 1, part=seg, cand=pool[best], fl=rfl(pool[best]),
                          t0=float(c.loc[cidxs]['t'].min()), t1=float(c.loc[cidxs]['t'].max()), idxs=cidxs))
        rem &= ~cm[best]; cm = np.delete(cm, best, axis=0); pool.pop(best)

# ---- 3架调度 ----
def tchg(s):
    if s < 0.90: return RTCHG * (0.65 * (0.90 - s) / 0.90 + 0.35)
    return RTCHG * 0.35 * (1.0 - s) / 0.10
rids = ['R01', 'R02', 'R03']; avail = {r: 0.0 for r in rids}
comps = {f'C#{i}': 0.0 for i in range(1, 7)}
ok = []
for w in sorted(tasks, key=lambda x: x['t0']):
    Eh = (RPH + RPC) * (w['t1'] - w['t0']) / 3600.0
    Et = w['fl']['Eg'] + w['fl']['Eb'] + Eh
    if Et > RBAT * (1 - RSOC): continue
    if 1 - Et / RBAT < RSOC: continue
    tp = max(0.0, w['t0'] - (RPREP + RLINK + w['fl']['tg']))
    r = min(rids, key=lambda x: avail[x])
    prep = max(tp, avail[r])
    sv = prep + RPREP + RLINK + w['fl']['tg']
    if sv > w['t1']: continue
    ch = None
    for cid in sorted(comps, key=lambda x: comps[x]):
        if comps[cid] <= prep: ch = cid; break
    if ch is None: continue
    back = w['t1'] + w['fl']['tb'] + RTURN
    w.update(Et=Et, rid=r, comp=ch, prep=prep, sv=sv, back=back)
    avail[r] = back; comps[ch] = back + tchg(1 - Et / RBAT)
    ok.append(w)
cov = set()
for w in ok:
    for i in w['idxs']:
        p = need.loc[i]
        if w['sv'] - 1e-9 <= p['t'] <= w['t1'] + 1e-9:
            cov.add(i)
nok = int(df['直连'].sum()) + len(cov)
print(f"[复核] 中继任务 {len(ok)}，需中继覆盖 {len(cov)}/{len(need)} = {len(cov)/len(need)*100:.1f}%")
print(f"[复核] 任务通信保障率 {nok}/{len(df)} = {nok/len(df)*100:.1f}%，残余中断 {len(need)-len(cov)} 点")
print(f"[复核] 中继能耗 {sum(w['Et'] for w in ok):.4f} kWh，联合总能耗 {65.25 + sum(w['Et'] for w in ok):.4f} kWh")
covset = set(cov)
per = {}
for fid in df['架次'].unique():
    n = len(df[df['架次'] == fid])
    nd = int(df[df['架次'] == fid]['直连'].sum())
    nc = sum(1 for i in need[need['架次'] == fid].index if i in covset)
    per[fid] = int(100 * (nd + nc) / n)
print("[复核] 100%保障架次数:", sum(1 for v in per.values() if v == 100), "/", len(per))
json.dump(dict(总采样=len(df), 直连点=int(df['直连'].sum()), 需中继=len(need), 中继任务=len(ok),
               需中继覆盖=len(cov), 保障率=round(nok / len(df) * 100, 1), 中断点=int(len(need) - len(cov)),
               中继能耗=round(sum(w['Et'] for w in ok), 4), 联合总能耗=round(65.25 + sum(w['Et'] for w in ok), 4)),
          open(BASE + r'\Q3_复核结果.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[复核] 结果写入 Q3_复核结果.json')
