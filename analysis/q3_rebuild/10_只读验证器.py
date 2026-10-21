# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成。
"""
Q3 只读验证器（pre-submission review P1 整改项）

目的：对冻结的中继方案做“只读”可行性核验——只读取已冻结的计划文件，
不搜索坐标、不重选悬停点、不把首段到位时刻直接赋值为窗口起点。

核验内容
  A 覆盖：按 (任务ID, 时段) 读取已冻结悬停点坐标，逐点重算双向链路；
     同时给出两种地形口径（HS=6m+双线性 / HS=0+最近像元）的通过率与最弱余量，
     用于量化“需求侧与核验侧口径差异”的影响。
  B 到位链：到位(建链完成) = 准备开始 + 180 + 航段时间 + 30，与首个需求时刻比较；
     航段时间按 DEM 巡航规则独立重算（V=15, VUP=4, VDN=3, 巡航高=最高地形+50）。
  C 转场净空：逐航段按 DEM 采样，检查巡航高与地形之间净空是否 >= 6m，
     并比较重算航段时间与冻结时刻表记录的段前转场时间之差。
  D 资源：中继机占用 [准备开始, 周转完成] 互斥、组件占用 [准备开始, 充电完成] 互斥、
     充电时长 = 500 + 1300*(1-SOC)、返航 SOC >= 0.20。

输出
  outputs_B/Q3_只读验证.json           汇总（含逐项 PASS/FAIL）
  outputs_B/Q3_只读验证_逐段.csv        分段覆盖结果（两种口径）
  outputs_B/Q3_只读验证_航段.csv        航段净空与时间差

不修改任何冻结计划文件；若发现违反，如实输出 FAIL 与数值。
"""
import os, json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from math import radians, sin, cos, sqrt, atan2, log10

_REPO_DEFAULT = r'c:\Users\Administrator\AppData\Local\Temp\论文投稿改造\repo\uav-flood-transport\analysis'
BASE = os.environ.get('Q3_REPO', _REPO_DEFAULT)
OUT = os.environ.get('Q3_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs_B'))
os.makedirs(OUT, exist_ok=True)
DEM_MAT = BASE + r'\数据\镇龙乡地理空间数据\镇龙乡及周边地理数据\数字高程模型数据（DEM）\镇龙乡及周边30米DEM.mat'

NS_PROF = 30
P_THR, C_FSPL = -90.0, 32.45
HS_MAIN, HS_ALT = 6.0, 0.0
R_V, R_VUP, R_VDN = 15.0, 4.0, 3.0
T_PREP, T_LINK, T_TURN = 180.0, 30.0, 300.0
CHG_FIX, CHG_SLOPE = 500.0, 1300.0
SOC_MIN = 0.20

_m = loadmat(DEM_MAT)
dem = np.array(_m['dem'], float)
lons = np.array(_m['longitude']).ravel(); lats = np.array(_m['latitude']).ravel()
dem[dem < 0] = np.nan
dlon = lons[1]-lons[0]; dlat0 = lats[0]-lats[1]

def hav_m(lo1, la1, lo2, la2):
    a = np.sin((la2-la1)/2.0)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2.0)**2
    return 6371000.0*2*np.arctan2(np.sqrt(a), np.sqrt(1-a))

def hav(lo1, la1, lo2, la2):
    return hav_m(radians(lo1), radians(la1), radians(lo2), radians(la2))

def at_nn(lo, la):
    i = np.rint((lats[0]-la)/dlat0).astype(int)
    j = np.rint((lo-lons[0])/dlon).astype(int)
    ok = (i >= 0) & (i < len(lats)) & (j >= 0) & (j < len(lons))
    out = np.full(np.shape(lo), np.nan)
    out[ok] = dem[np.clip(i[ok], 0, len(lats)-1), np.clip(j[ok], 0, len(lons)-1)]
    return out

def at_bi(lo, la):
    j = (lo-lons[0])/dlon; i = (lats[0]-la)/dlat0
    j0 = np.floor(j).astype(int); i0 = np.floor(i).astype(int)
    ok = (j0 >= 0) & (i0 >= 0) & (j0 <= len(lons)-2) & (i0 <= len(lats)-2)
    out = np.full(np.shape(lo), np.nan)
    if not ok.any():
        return out
    fj = j[ok]-j0[ok]; fi = i[ok]-i0[ok]
    a = dem[i0[ok], j0[ok]]; b = dem[i0[ok], j0[ok]+1]
    c = dem[i0[ok]+1, j0[ok]]; d = dem[i0[ok]+1, j0[ok]+1]
    out[ok] = (1-fj)*(1-fi)*a + fj*(1-fi)*b + (1-fj)*fi*c + fj*fi*d
    return out

def blocked(P, C, hs, interp):
    """P:(n,3) 需求点, C:(1,3) 悬停点；返回 (n,) 是否遮挡。"""
    terr = at_nn if interp == 'nn' else at_bi
    ok = np.ones(len(P), bool)
    for k in range(1, NS_PROF+1):
        u = k/(NS_PROF+1)
        lo = P[:, 0] + u*(C[0, 0]-P[:, 0])
        la = P[:, 1] + u*(C[0, 1]-P[:, 1])
        ht = terr(lo, la)
        hl = P[:, 2] + u*(C[0, 2]-P[:, 2])
        ok &= ~(ht + hs > hl)
    return ~ok

def link_margin(P, C, hs, interp):
    """返回两点各方向的弱向 Pr (dBm)：TR->relay 与 relay->G01。"""
    d2 = hav_m(np.radians(P[:, 0]), np.radians(P[:, 1]),
               radians(C[0, 0]), radians(C[0, 1]))
    d3 = np.sqrt(d2**2 + (P[:, 2]-C[0, 2])**2)
    pr_tr = 26.0 - (C_FSPL + 20*log10(2400.0) + 20*np.log10(d3/1000.0)
                    + 10.0*blocked(P, C, hs, interp))
    dg = hav_m(radians(G01[0]), radians(G01[1]), radians(C[0, 0]), radians(C[0, 1]))
    d3g = np.sqrt(dg**2 + (G01[2]-C[0, 2])**2)
    pr_g01 = 36.0 - (C_FSPL + 20*log10(2400.0) + 20*np.log10(d3g/1000.0)
                     + 10.0*blocked_g01(C, hs, interp))
    return pr_tr, pr_g01

def blocked_g01(C, hs, interp):
    terr = at_nn if interp == 'nn' else at_bi
    for k in range(1, NS_PROF+1):
        u = k/(NS_PROF+1)
        lo = np.array([G01[0] + u*(C[0, 0]-G01[0])])
        la = np.array([G01[1] + u*(C[0, 1]-G01[1])])
        ht = terr(lo, la)[0]
        hl = G01[2] + u*(C[0, 2]-G01[2])
        if np.isfinite(ht) and (ht + hs > hl):
            return 1.0
    return 0.0

def cruise_alt(a, b):
    """Q3_TD-CJO_v2 口径：网格边界交点+相邻中点采样取最高高程+50m。"""
    ts = [0.0, 1.0]
    for axis, x1, x2 in [(lons, a[0], b[0]), (lats, a[1], b[1])]:
        bd = (axis[:-1]+axis[1:])/2.0
        if abs(x2-x1) < 1e-12:
            continue
        t = (bd-x1)/(x2-x1)
        ts += t[(t > 0) & (t < 1)].tolist()
    ts = sorted(set(ts))
    sp = [0.0, 1.0] + [(u+v)/2 for u, v in zip(ts, ts[1:])]
    return float(np.nanmax(at_nn(np.array([a[0]+t*(b[0]-a[0]) for t in sp]),
                                np.array([a[1]+t*(b[1]-a[1]) for t in sp]))) + 50.0)

def leg_time_clear(a, b):
    """返回 (飞行时间s, 巡航高m, 最小净空m)。a/b = (lon, lat, alt)。"""
    d = hav(a[0], a[1], b[0], b[1])
    hc = cruise_alt(a, b)
    t_cl = max(hc-a[2], 0.0)/R_VUP
    t_cr = d/R_V
    t_dn = max(hc-b[2], 0.0)/R_VDN
    lo = np.linspace(a[0], b[0], 200); la = np.linspace(a[1], b[1], 200)
    terr = at_nn(lo, la)
    clear = hc - np.nanmax(terr)
    return t_cl+t_cr+t_dn, hc, float(clear)

nodes = pd.read_csv(BASE + r'\预处理输出\Q1_节点_预处理.csv')
_o = nodes[nodes['编号'] == 'O01'].iloc[0]
O = (float(_o['经度']), float(_o['纬度']), float(_o['海拔(m)']))
G01 = (O[0], O[1], O[2] + 20.0)

tt = pd.read_csv(os.path.join(OUT, 'Q3_固定方案_悬停点时窗.csv'), encoding='utf-8-sig')
pts = pd.read_csv(os.path.join(OUT, 'Q3_规范集合_覆盖核验逐点.csv'), encoding='utf-8-sig')
sched = pd.read_csv(BASE + r'\Q3_两架方案_中继排程.csv', encoding='utf-8-sig')
soc = pd.read_csv(os.path.join(OUT, 'Q3_核验方案_能耗SOC对照_eta0.72.csv'), encoding='utf-8-sig')

# ---------- A 覆盖 ----------
seg_rows = []
for _, r in tt.iterrows():
    mid, si = r['任务ID'], int(r['时段'])
    sub = pts[(pts['保障中继'] == mid) & (pts['时段'] == si)]
    P = sub[['lon', 'lat', 'h']].values.astype(float)
    C = np.array([[float(r['经度']), float(r['纬度']), float(r['海拔m'])]])
    for tag, hs, interp in [('HS6_bilinear', HS_MAIN, 'bi'), ('HS0_nearest', HS_ALT, 'nn')]:
        pra, prg = link_margin(P, C, hs, interp)
        ok = (pra >= P_THR) & (prg >= P_THR)
        seg_rows.append(dict(任务ID=mid, 时段=si, 口径=tag, 点数=len(P),
                             通过=int(ok.sum()), 通过率=round(float(ok.mean()), 6),
                             最弱_TR中继Pr=round(float(pra.min()), 2),
                             中继G01_Pr=round(float(prg), 2),
                             最小余量dB=round(float(min((pra-P_THR).min(), prg-P_THR)), 2)))
seg = pd.DataFrame(seg_rows)
seg.to_csv(os.path.join(OUT, 'Q3_只读验证_逐段.csv'), index=False, encoding='utf-8-sig')

# ---------- B/C 航段 ----------
chain = {}
for _, r in sched.iterrows():
    chain[r['任务ID']] = [(float(r['经度']), float(r['纬度']), float(r['海拔m']))]
for _, r in tt.sort_values(['任务ID', '时段']).iterrows():
    p = (float(r['经度']), float(r['纬度']), float(r['海拔m']))
    if p not in chain[r['任务ID']]:
        chain[r['任务ID']].append(p)

leg_rows = []
for _, r in sched.iterrows():
    mid = r['任务ID']
    prep = float(r['准备开始s']); land = float(r['返航落地s'])
    pts_chain = chain[mid]
    legs = [('出航', O, pts_chain[0])] + \
           [('段间%d' % (i+1), pts_chain[i], pts_chain[i+1]) for i in range(len(pts_chain)-1)] + \
           [('返航', pts_chain[-1], O)]
    for name, a, b in legs:
        t_fly, hc, clear = leg_time_clear(a, b)
        leg_rows.append(dict(任务ID=mid, 航段=name, 距离m=round(hav(a[0], a[1], b[0], b[1]), 0),
                             巡航高m=round(hc, 1), 最小净空m=round(clear, 1),
                             重算飞行s=round(t_fly, 1), 净空通过=clear >= HS_MAIN))
    t_fly_out = leg_time_clear(O, pts_chain[0])[0]
    ready = prep + T_PREP + t_fly_out + T_LINK
    first_t = float(tt[(tt['任务ID'] == mid)].sort_values('时段').iloc[0]['在岗起s'])
    leg_rows.append(dict(任务ID=mid, 航段='首段到位', 距离m=np.nan, 巡航高m=np.nan,
                         最小净空m=np.nan, 重算飞行s=round(ready, 1),
                         净空通过=ready <= first_t))

# 段间转场：冻结时刻表与重算之差
inter = []
for _, r in tt[tt['段前转场s'].notna()].iterrows():
    mid, si = r['任务ID'], int(r['时段'])
    prev = tt[(tt['任务ID'] == mid) & (tt['时段'] == si-1)].iloc[0]
    a = (float(prev['经度']), float(prev['纬度']), float(prev['海拔m']))
    b = (float(r['经度']), float(r['纬度']), float(r['海拔m']))
    t_fly, hc, clear = leg_time_clear(a, b)
    inter.append(dict(任务ID=mid, 时段=si, 冻结转场s=float(r['段前转场s']),
                      重算转场s=round(float(t_fly), 1), 差s=round(float(t_fly-float(r['段前转场s'])), 1),
                      巡航高m=round(float(hc), 1), 最小净空m=round(float(clear), 1),
                      净空通过=bool(clear >= HS_MAIN),
                      到位可行=bool(float(r['到位时刻s']) >= float(prev['在岗止s']) + t_fly + T_LINK)))
legs = pd.DataFrame(leg_rows)
legs.to_csv(os.path.join(OUT, 'Q3_只读验证_航段.csv'), index=False, encoding='utf-8-sig')

# ---------- D 资源 ----------
def overlaps(iv):
    iv = sorted(iv)
    return sum(1 for (s1, e1), (s2, e2) in zip(iv, iv[1:]) if s2 < e1)

mach = {}
comp = {}
for _, r in sched.iterrows():
    mach.setdefault(str(r['中继编号']), []).append((float(r['准备开始s']), float(r['周转完成s'])))
    comp.setdefault(str(r['能源组件']), []).append((float(r['准备开始s']), float(r['充电完成s'])))
mach_bad = sum(overlaps(v) for v in mach.values())
comp_bad = sum(overlaps(v) for v in comp.values())

chg_rows = []
for _, r in soc.iterrows():
    mid = r['任务ID']
    s = sched[sched['任务ID'] == mid].iloc[0]
    land = float(s['返航落地s']); chg = float(s['充电完成s'])
    soc_frozen = float(s['返航SOC']); soc_v = float(r['返航SOC_核验方案'])
    pred = CHG_FIX + CHG_SLOPE*(1.0-soc_frozen)
    chg_rows.append(dict(任务ID=mid, 返航SOC_冻结=soc_frozen, 返航SOC_重算=soc_v,
                         SOC差=round(soc_v-soc_frozen, 4),
                         冻结充电s=round(chg-land, 1), 公式预测s=round(pred, 1),
                         差s=round(chg-land-pred, 1),
                         充电公式通过=abs(chg-land-pred) <= 1.0, SOC达标=soc_v >= SOC_MIN))

# ---------- E 口径差异的修复建议（只出建议，不修改冻结计划） ----------
# 对在 HS=6m+双线性 下未全覆盖的分段，在冻结点 ±0.03° 内按 0.0025° 步长搜索
# “离冻结点最近且能全覆盖”的候选（离地 300m、距 O01<=12km），并给出该候选的
# 段前转场、到岗余量与架次级能耗/SOC（能耗口径与脚本07一致，eta_c=0.72）。
ETA_CL, P_CR, P_HOV, E_BAT, MASS, G_ACC, O_ALT = 0.72, 1.15, 1.10, 3.2, 23.5, 9.81, 127.7

def sortie_energy(chain_pts, srv, t_trans):
    """与脚本07相同口径：E转场=P_CR*T_fly/3600 + m*g*ΣΔh_爬升/(eta*3.6e6)。"""
    first, last = chain_pts[0], chain_pts[-1]
    t_fly = max(first[2]-O_ALT, 0.0)/R_VUP + \
            hav_m(radians(O[0]), radians(O[1]), radians(first[0]), radians(first[1]))/R_V
    dh_up = max(first[2]-O_ALT, 0.0)
    for a, b in zip(chain_pts, chain_pts[1:]):
        dh = b[2]-a[2]
        t_fly += (dh/R_VUP if dh >= 0 else -dh/R_VDN) + \
                 hav_m(radians(a[0]), radians(a[1]), radians(b[0]), radians(b[1]))/R_V
        if dh > 0:
            dh_up += dh
    t_fly += hav_m(radians(last[0]), radians(last[1]), radians(O[0]), radians(O[1]))/R_V + \
             max(last[2]-O_ALT, 0.0)/R_VDN
    e_fly = P_CR*t_fly/3600.0 + G_ACC*MASS*dh_up/(ETA_CL*3.6e6)
    e_tot = e_fly + P_HOV*(srv-t_trans)/3600.0
    return e_tot, 1.0-e_tot/E_BAT

repair_rows = []
for _, r in seg[(seg['口径'] == 'HS6_bilinear') & (seg['通过率'] < 1)].iterrows():
    mid, si = r['任务ID'], int(r['时段'])
    sub = pts[(pts['保障中继'] == mid) & (pts['时段'] == si)]
    P = sub[['lon', 'lat', 'h']].values.astype(float)
    _row = tt[(tt['任务ID'] == mid) & (tt['时段'] == si)].iloc[0]
    clon, clat, calt = float(_row['经度']), float(_row['纬度']), float(_row['海拔m'])
    best = None
    for dlo in np.arange(-0.03, 0.0301, 0.0025):
        for dla in np.arange(-0.03, 0.0301, 0.0025):
            lo, la = clon+dlo, clat+dla
            if hav(lo, la, O[0], O[1]) > 12000.0:
                continue
            e = float(at_bi(np.array([lo]), np.array([la]))[0])
            if not np.isfinite(e):
                continue
            cand = np.array([[lo, la, e+300.0]])
            pra, prg = link_margin(P, cand, HS_MAIN, 'bi')
            if not bool(((pra >= P_THR) & (prg >= P_THR)).all()):
                continue
            d = hav(lo, la, clon, clat)
            if best is None or d < best[0]:
                best = (d, cand, float(pra.min()), float(prg))
    if best is None:
        repair_rows.append(dict(任务ID=mid, 时段=si, 结论='±0.03°内无全覆盖候选'))
        continue
    dmin, cand, pra_min, prg_min = best
    rows_mid = tt[tt['任务ID'] == mid].sort_values('时段').reset_index(drop=True)
    idx = int(rows_mid.index[rows_mid['时段'] == si][0])
    chain_new = [(float(x['经度']), float(x['纬度']), float(x['海拔m']))
                 for _, x in rows_mid.iterrows()]
    chain_new[idx] = (float(cand[0, 0]), float(cand[0, 1]), float(cand[0, 2]))
    t_trans = sum(leg_time_clear(a, b)[0] for a, b in zip(chain_new, chain_new[1:]))
    arrive = float(rows_mid.iloc[idx-1]['在岗止s']) + \
             leg_time_clear(chain_new[idx-1], chain_new[idx])[0] + T_LINK if idx > 0 else np.nan
    _s = sched[sched['任务ID'] == mid].iloc[0]
    srv = float(_s['离岗时刻s']) - float(_s['到岗时刻s'])   # 与脚本07一致：含首段前在岗等待
    e_tot, soc_v = sortie_energy(chain_new, srv, t_trans)
    repair_rows.append(dict(任务ID=mid, 时段=si, 冻结点='(%.5f, %.5f, %.1f)' % (clon, clat, calt),
                            位移m=round(float(dmin), 0),
                            修复点='(%.5f, %.5f, %.1f)' % (cand[0, 0], cand[0, 1], cand[0, 2]),
                            最弱TR中继Pr=round(pra_min, 2), 中继G01_Pr=round(prg_min, 2),
                            冻结转场s=float(_row['段前转场s']), 重算转场s=round(t_trans, 1),
                            到岗时刻s=round(arrive, 1), 窗口起点s=float(_row['在岗起s']),
                            到位余量s=round(float(_row['在岗起s'])-arrive, 1),
                            架次能耗kWh=round(e_tot, 4), 冻结能耗kWh=float(_s['架次能耗kWh']),
                            返航SOC=round(soc_v, 3), SOC达标=bool(soc_v >= SOC_MIN)))
rep = pd.DataFrame(repair_rows)
rep.to_csv(os.path.join(OUT, 'Q3_只读验证_修复建议.csv'), index=False, encoding='utf-8-sig')

# ---------- 汇总 ----------
res = {}
for tag in ['HS6_bilinear', 'HS0_nearest']:
    g = seg[seg['口径'] == tag]
    res[tag] = dict(点数=int(g['点数'].sum()), 通过=int(g['通过'].sum()),
                    通过率=round(float(g['通过'].sum()/g['点数'].sum()), 6),
                    最小余量dB=round(float(g['最小余量dB'].min()), 2))
_first = legs[legs['航段'] == '首段到位'].set_index('任务ID')['重算飞行s']
_win0 = tt.sort_values(['任务ID', '时段']).groupby('任务ID')['在岗起s'].min()
_margin = (_win0 - _first).round(1)
res['首段到位余量s'] = {k: float(v) for k, v in _margin.items()}
res['首段到位最迟延误s'] = float(max(0.0, -_margin.min()))
res['航段净空全部通过'] = bool(legs[legs['航段'] != '首段到位']['净空通过'].all())
res['最小航段净空m'] = float(legs[legs['航段'] != '首段到位']['最小净空m'].min())
res['段间转场时间差s'] = [dict(x) for x in inter]
res['中继机占用冲突'] = int(mach_bad)
res['组件占用冲突'] = int(comp_bad)
res['充电公式'] = [dict(x) for x in chg_rows]
res['充电公式全部通过'] = all(x['充电公式通过'] for x in chg_rows)
res['SOC全部达标'] = all(x['SOC达标'] for x in chg_rows)
res['修复建议'] = [dict(x) for x in repair_rows]

with open(os.path.join(OUT, 'Q3_只读验证.json'), 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

print(seg.to_string(index=False))
print()
print(legs.to_string(index=False))
print()
print(pd.DataFrame(inter).to_string(index=False))
print()
print(pd.DataFrame(chg_rows).to_string(index=False))
print()
print(json.dumps({k: v for k, v in res.items() if k not in ('段间转场时间差s', '充电公式')},
                 ensure_ascii=False, indent=2))
