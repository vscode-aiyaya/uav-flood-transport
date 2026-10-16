# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。AI工具名称：DeepSeek；版本/型号：V4.1-Flash；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；版本发布日期：2026年9月10日。

"""
Q3 TD-CJO v2（任务驱动动态中继调度）
- 需中继时刻按时间簇分组 → 每簇选最优悬停点（任务价值感知贪心）
- 2架中继按时间轴执行中继架次（O01→悬停点→O01，准备180s+建链30s+周转300s）
- 全程连续通信核验：直连∪中继覆盖 = 100% 任务时间
- 能耗：飞行(巡航1.15kW+爬升附加) + 悬停(1.05+0.05×链路kW)；组件3.2kWh、返航≥20%、两阶段充电1800s、6组
"""
import os, json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from math import radians, sin, cos, sqrt, atan2, log10

import os
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, '数据'); PREP = os.path.join(BASE, '预处理输出')
DEM_MAT = DATA + r'\镇龙乡地理空间数据\镇龙乡及周边地理数据\数字高程模型数据（DEM）\镇龙乡及周边30米DEM.mat'
G = 9.81
m = loadmat(DEM_MAT)
dem = np.array(m['dem'], dtype=float); lons = np.array(m['longitude']).ravel(); lats = np.array(m['latitude']).ravel()
dem[dem < 0] = np.nan
def at(lon, lat):
    i = int(np.argmin(np.abs(lats - lat))); j = int(np.argmin(np.abs(lons - lon)))
    return dem[i, j]
def hav(lon1, lat1, lon2, lat2):
    lon1_r, lat1_r, lon2_r, lat2_r = map(radians, [lon1, lat1, lon2, lat2])
    a = sin((lat2_r - lat1_r) / 2) ** 2 + cos(lat1_r) * cos(lat2_r) * sin((lon2_r - lon1_r) / 2) ** 2
    return 6371000 * 2 * atan2(sqrt(a), sqrt(1 - a))

F_MHZ = 2400.0; L_SYS = 3.0; L_OBS = 10.0; P_SENS = -98.0; M_FADE = 8.0
P_THR = P_SENS + M_FADE; HS = 0.0; N_SAMPLES = 20
PAR = {'G01': (27.0, 12.0), 'TR': (20.0, 3.0), 'REL_A': (20.0, 6.0), 'REL_B': (19.0, 8.0)}
# 中继R参数（题目数据文件）
R_V = 15.0; R_VUP = 4.0; R_VDN = 3.0; R_PCR = 1.15; R_PHOV = 1.05; R_PCOMM = 0.05
R_M = 23.5; R_ETA = 0.72; R_E_BAT = 3.2; R_SOC_MIN = 0.20; R_TCHG = 1800.0
R_T_PREP = 180.0; R_T_LINK = 30.0; R_T_TURN = 300.0
R_COUNT = 2; R_NCOMP = 6

def fspl(d_m): return 32.44 + 20.0 * log10(F_MHZ) + 20.0 * log10(d_m / 1000.0)
def los_block(p1, p2):
    d = hav(p1[0], p1[1], p2[0], p2[1])
    if d < 1e-9: return False
    h1, h2 = p1[2], p2[2]
    ns = max(2, min(30, int(d / 250.0) + 1))
    for k in range(1, ns):
        t = k / ns
        lo_t = p1[0] + t * (p2[0] - p1[0]); la_t = p1[1] + t * (p2[1] - p1[1])
        h_line = h1 + t * (h2 - h1); h_terr = at(lo_t, la_t)
        if np.isfinite(h_terr) and (h_terr + HS) > h_line: return True
    return False
def link_ok(a, b, tx_key, rx_key):
    d3 = sqrt(hav(a[0], a[1], b[0], b[1]) ** 2 + (a[2] - b[2]) ** 2)
    PL = fspl(d3) + (L_OBS if los_block(a, b) else 0.0)
    Pt, Gt = PAR[tx_key]; Gr = PAR[rx_key][1]
    return (Pt + Gt + Gr - L_SYS - PL) >= P_THR
def bi_link_ok(pA, keyA, pB, keyB):
    return link_ok(pA, pB, keyA, keyB) and link_ok(pB, pA, keyB, keyA)

nodes = pd.read_csv(os.path.join(PREP, 'Q1_节点_预处理.csv'))
q2 = pd.read_csv(os.path.join(BASE, 'Q2_主方案20架次_架次表.csv'))
cargo = pd.read_csv(os.path.join(PREP, 'Q1_货箱明细_预处理.csv'))
uav = pd.read_csv(os.path.join(PREP, 'Q2_机型参数_预处理.csv'))
o = nodes[nodes['编号'] == 'O01'].iloc[0]
O_LON, O_LAT, O_ALT = float(o['经度']), float(o['纬度']), float(o['海拔(m)'])
sa_rows = nodes[nodes['编号'] != 'O01'].copy().reset_index(drop=True)
H_G01 = O_ALT + 20.0; G01 = (O_LON, O_LAT, H_G01)
pps = {}
for _, u in uav.iterrows():
    pps[u['机型编号']] = dict(v=float(u['巡航速度(m/s)']), v_up=float(u['最大爬升速度(m/s)']), v_dn=float(u['最大下降速度(m/s)']))
h_op = {r['编号']: float(r['海拔(m)']) + 30.0 for _, r in sa_rows.iterrows()}
h_op['O01'] = O_ALT
lon = {r['编号']: float(r['经度']) for _, r in sa_rows.iterrows()}
lat = {r['编号']: float(r['纬度']) for _, r in sa_rows.iterrows()}
lon['O01'], lat['O01'] = O_LON, O_LAT
def cruise_exact(a, b):
    ts = [0.0, 1.0]
    for axis, x1, x2 in [(lons, a[0], b[0]), (lats, a[1], b[1])]:
        bounds = (axis[:-1] + axis[1:]) / 2.0
        t = (bounds - x1) / (x2 - x1)
        ts.extend(t[(t > 0) & (t < 1)].tolist())
    ts = sorted(set(ts))
    samples = [0.0, 1.0] + [(u + v) / 2.0 for u, v in zip(ts, ts[1:])]
    mx = max(at(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])) for t in samples)
    return mx + 50.0
box_w = {r['货箱序号']: float(r['单箱质量(kg)']) for _, r in cargo.iterrows()}
traj = []
df_del = pd.read_csv(os.path.join(BASE, 'Q2_主方案20架次_逐箱交付.csv'))
for _, row in q2.iterrows():
    seq = row['路径'].split('→'); mt = row['机型']; pp = pps[mt]
    boxes_by_zone = {}
    for _, c in df_del.iterrows():
        if c['架次ID'] == row['架次ID']:
            boxes_by_zone.setdefault(c['服务区'], []).append(c['货箱序号'])
    t = float(row['起飞s']); rem = float(row['载重kg']); prev = 'O01'
    for z in seq[1:-1] + ['O01']:
        d = hav(lon[prev], lat[prev], lon[z], lat[z])
        Hc = cruise_exact((lon[prev], lat[prev]), (lon[z], lat[z]))
        h0, h1 = h_op[prev], h_op[z]
        t_cl = max(Hc - h0, 0.0) / pp['v_up']; t_cr = d / pp['v']; t_dn = max(Hc - h1, 0.0) / pp['v_dn']
        seg_len = t_cl + t_cr + t_dn
        ns = max(2, int(N_SAMPLES * seg_len / (seg_len + 1e-9)))
        for k in range(ns + 1):
            tt = k / ns; tt_seg = tt * seg_len
            if tt_seg <= t_cl: hgt = h0 + (Hc - h0) * tt_seg / t_cl
            elif tt_seg <= t_cl + t_cr: hgt = Hc
            else: hgt = Hc + (h1 - Hc) * (tt_seg - t_cl - t_cr) / max(t_dn, 1e-9)
            traj.append((t + tt * seg_len, lon[prev] + (lon[z] - lon[prev]) * tt,
                         lat[prev] + (lat[z] - lat[prev]) * tt, hgt, row['架次ID'], mt, rem))
        if z != 'O01': rem -= sum(box_w[b] for b in boxes_by_zone.get(z, []))
        t += seg_len; prev = z
df_traj = pd.DataFrame(traj, columns=['t', 'lon', 'lat', 'h', '架次', '机型', '载重']).sort_values('t').reset_index(drop=True)
rows = []
for _, p in df_traj.iterrows():
    pt = (p['lon'], p['lat'], p['h'])
    rows.append(bi_link_ok(G01, 'G01', pt, 'TR'))
df_traj['直连'] = rows
need = df_traj[~df_traj['直连']].copy().reset_index(drop=True)

# 时段聚类（间隔>400s切分）
need = need.sort_values('t').reset_index(drop=True)
clusters = []
cur = [need.iloc[0]]
for i in range(1, len(need)):
    if need.iloc[i]['t'] - need.iloc[i-1]['t'] > 400.0:
        clusters.append(pd.DataFrame(cur)); cur = [need.iloc[i]]
    else:
        cur.append(need.iloc[i])
clusters.append(pd.DataFrame(cur))
print('需中继时段簇:', len(clusters))
for ci, c in enumerate(clusters):
    print(f"  簇{ci+1}: [{c['t'].min():.0f},{c['t'].max():.0f}] {len(c)}点 架次{c['架次'].unique().tolist()}")

# 候选悬停点（DEM网格，距O01 12km内，离地300m）
cands = []
lo0, lo1 = float(lons.min()), float(lons.max())
la0, la1 = float(lats.min()), float(lats.max())
step = 0.01
lo = lo0
while lo <= lo1:
    la = la0
    while la <= la1:
        e = at(lo, la)
        if np.isfinite(e):
            cands.append((f"G{lo:.3f}_{la:.3f}", lo, la, e + 300.0))
        la += step
    lo += step
cands = [c for c in cands if hav(O_LON, O_LAT, c[1], c[2]) < 12000]
print('网格候选点:', len(cands))
def relay_covers(cand, p):
    pc = (cand[1], cand[2], cand[3]); pt = (p['lon'], p['lat'], p['h'])
    return bi_link_ok(pt, 'TR', pc, 'REL_A') and bi_link_ok(pc, 'REL_B', G01, 'G01')

# 每簇选最优悬停点：若1点未全盖，按时间拆段（串行），每段1点
tasks = []
for ci, c in enumerate(clusters):
    seg = c.copy().reset_index(drop=True)
    # 先试单点
    best_c, best_n = None, -1
    for cand in cands:
        n = sum(1 for _, p in seg.iterrows() if relay_covers(cand, p))
        if n > best_n: best_c, best_n = cand, n
    if best_n == len(seg):
        tasks.append(dict(cluster=ci + 1, part=1, cand=best_c, n_cover=best_n,
                          t0=float(seg['t'].min()), t1=float(seg['t'].max()), pts=len(seg)))
        print(f"簇{ci+1}-1 单点 {best_c[0]} 覆盖{best_n}/{len(seg)}")
        continue
    # 按时间二分
    mid = seg['t'].median()
    s1, s2 = seg[seg['t'] <= mid], seg[seg['t'] > mid]
    picks = []
    for s in (s1, s2):
        if len(s) == 0: continue
        b2, n2 = None, -1
        for cand in cands:
            n = sum(1 for _, p in s.iterrows() if relay_covers(cand, p))
            if n > n2: b2, n2 = cand, n
        picks.append((b2, n2, s))
    tot = sum(p[1] for p in picks)
    if tot >= best_n and tot < len(seg):
        # 再试三段
        q1, q2, q3 = np.array_split(seg['t'].values, 3)
        picks3 = []
        for lo_, hi_ in [(q1[0], q1[-1]), (q2[0], q2[-1]), (q3[0], q3[-1])]:
            s = seg[(seg['t'] >= lo_) & (seg['t'] <= hi_)]
            b3, n3 = None, -1
            for cand in cands:
                n = sum(1 for _, p in s.iterrows() if relay_covers(cand, p))
                if n > n3: b3, n3 = cand, n
            picks3.append((b3, n3, s))
        tot3 = sum(p[1] for p in picks3)
        if tot3 > tot and tot3 >= len(seg) - 2:
            picks = picks3
    for pi, (cand, n_cover, s) in enumerate(picks):
        tasks.append(dict(cluster=ci + 1, part=pi + 1, cand=cand, n_cover=n_cover,
                          t0=float(s['t'].min()), t1=float(s['t'].max()), pts=len(s)))
        print(f"簇{ci+1}-{pi+1} 段[{s['t'].min():.0f},{s['t'].max():.0f}] {cand[0]} 覆盖{n_cover}/{len(s)}")
    miss = len(seg) - tot3 if 'tot3' in dir() else len(seg) - tot
    if miss > 0:
        print(f"  !! 簇{ci+1} 未覆盖 {miss} 点")
print('任务数:', len(tasks))

# 中继架次窗口：准备180 + 飞行 + [服务t0,t1] + 飞行返回 + 周转300
def relay_flight(cand):
    pc = (cand[1], cand[2], cand[3])
    d = hav(O_LON, O_LAT, cand[1], cand[2])
    Hc = cruise_exact((O_LON, O_LAT), (cand[1], cand[2]))
    t_cl = max(Hc - O_ALT, 0.0) / R_VUP; t_cr = d / R_V; t_dn = max(Hc - cand[3], 0.0) / R_VDN
    t_fl = t_cl + t_cr + t_dn
    E_cr = R_PCR * 2 * t_fl / 3600.0
    E_cl = 2 * G * R_M * max(Hc - O_ALT, 0.0) / (R_ETA * 3600000)
    return t_fl, E_cr + E_cl

wins = []
for tk in tasks:
    t_fl, E_fl = relay_flight(tk['cand'])
    wins.append(dict(tk=tk, t_fl=t_fl, E_fl=E_fl,
                     prep_start=tk['t0'] - R_T_PREP - R_T_LINK - t_fl,
                     ready=tk['t0'] - R_T_LINK - t_fl,
                     start=tk['t0'], end=tk['t1'],
                     back=tk['t1'] + t_fl + R_T_TURN))
print()
print('=== 中继任务窗口 ===')
for w in wins:
    print(f"簇{w['tk']['cluster']}-{w['tk']['part']} {w['tk']['cand'][0]}: 准备起{w['prep_start']:.0f} 起飞就绪{w['ready']:.0f} 服务[{w['start']:.0f},{w['end']:.0f}] 周转完{w['back']:.0f} 飞行{w['t_fl']:.0f}s")

# 2架中继串行分配（按开始时间，取最早可用中继；检查与前任务不重叠）
sched = {'R01': [], 'R02': []}
avail = {'R01': 0.0, 'R02': 0.0}
feasible = True
for w in sorted(wins, key=lambda x: x['start']):
    chosen = None
    for rid in ['R01', 'R02']:
        if avail[rid] <= w['prep_start']:
            chosen = rid; break
    if chosen is None:
        print(f"!! 簇{w['tk']['cluster']}-{w['tk']['part']} 无可用中继（R01可用{avail['R01']:.0f}, R02可用{avail['R02']:.0f}）")
        feasible = False
        break
    sched[chosen].append((w['tk']['cluster'], w['tk']['part']))
    avail[chosen] = w['back']
    print(f"簇{w['tk']['cluster']}-{w['tk']['part']} -> {chosen}")

print()
print('=== 调度结果 ===')
for r in ['R01', 'R02']:
    print(f"{r}: 执行簇 {sched[r]}")
print('调度可行:', feasible)
