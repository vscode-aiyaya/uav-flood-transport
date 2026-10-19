# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成。
"""
Q3重建 脚本05：接收门限敏感性（论文5.4.2声称值的独立复算）

论文定义逐字引文（PDF 5.2.2 / 5.4.2 节）：
  [5.2.2] “对问题二20 架次运输轨迹按起降、巡航、交接阶段逐秒采样（dt = 1 s）并
          逐点判定与G01 的双向链路：主方案口径下识别需中继点11593 个（−90 dBm
          门限）……直连率、门限与安全超高对需中继点数量的敏感性见5.4.2 节
          （搜索口径dt = 5 s 下Hs = 0 时直连率60.2%、需中继点2368 个）。”
  [5.4.2] “（2）敏感性分析：接收门限−90/ −95/ −98 dBm 对应直连服务区3/10/15
          （图5.6d），需中继点2368/806/0（搜索口径dt = 5 s）；……
          DEM 安全超高Hs = 0/25/50 m 对应直连率60.2%/65.8%/74.8%（图5.6c）。”

本脚本按上述定义如实重算“搜索口径”（dt=5s、Hs=0）下三个门限对应的
需中继点数、直连率与直连服务区数，并附 Hs 扫描，输出实测值。
**不向 2368/806/0 反推任何参数**：所有物理常量与轨迹口径均取自脚本01/02
（题目附录可溯源部分），仅改变采样步长与门限。

【假设清单】（论文定义不完整处集中声明，均不影响“如实重算”的性质）：
  A1 “dt=5s”：各架次按相位以约5s步长采样，相位端点必取；
     每相位点数 n = max(1, int(时长/5))，步长 = 时长/n（≈5s）。
     不采用“1s轨迹按第5个点抽稀”，以免引入相位边界重复。
  A2 “Hs=0”：DEM 视线判定地形安全超高取 0 m（论文声称的主方案保守口径）。
     Hs 扫描存在两种读法（论文5.4.2 直连率随Hs升而不降，与“地形余量”方向相反），
     故并列报告：(A) 地形视线余量（加到地形上，Hs越大遮挡越多）；(B) 飞行安全
     超高（加到飞行器高度上，Hs越大视线越高）。二者均按论文给定 Hs=0/25/50 扫描。
  A3 DEM 插值口径：双线性（脚本01 已用冻结输出现地校准为该口径）。
  A4 直连判定：双向弱方向，EIRP+Gr−Lsys = 32 dB（G01 27+12 / 运输机 20+3），
     自由空间常数 32.45、系统损耗 3 dB、遮挡附加 10 dB；视线剖面 100 m 采样、
     单链路上限 60 点。
  A5 “直连服务区”：某服务区 z（S001–S015，取航段终点/交接区）若其全部采样点
     均直连成功，则计为该门限下的“直连服务区”，否则该区需中继。O01 端点不计入。
  A6 轨迹几何：完全沿用脚本01 的航段/巡航高度口径（含两项已声明的逆向锁定
     经验项：出航爬升+37s、O01–S007 用 H_center），仅改变采样步长与门限，
     以隔离“采样/门限”这一自变量。

输出：outputs\\Q3_重建_门限敏感性.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from math import radians, sin, cos, sqrt, atan2, log10

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_DEFAULT = os.path.normpath(os.path.join(_HERE, '..'))
BASE = os.environ.get('Q3_REPO', _REPO_DEFAULT)
OUT  = os.environ.get('Q3_OUT', os.path.join(_HERE, 'outputs'))
os.makedirs(OUT, exist_ok=True)
DEM_MAT = os.path.join(BASE, '数据', '镇龙乡地理空间数据', '镇龙乡及周边地理数据', '数字高程模型数据（DEM）', '镇龙乡及周边30米DEM.mat')

# ---- 链路常量（脚本01口径，题目附录可溯源） ----
F_MHZ, FSPL_C, L_SYS, L_OBS = 2400.0, 32.45, 3.0, 10.0
PR_TX_SUM = min(27.0+12.0+3.0-L_SYS, 20.0+3.0+12.0-L_SYS)   # = 32 dB 弱方向
TERR_STEP, TERR_CAP = 100.0, 60
SAMPLING_DT = 5.0                                            # 【A1】搜索口径 dt=5s
PTHRS = [-90.0, -95.0, -98.0]                                # 论文三档门限
HS_SCAN = [0.0, 25.0, 50.0]                                  # 论文 Hs 扫描

_m = loadmat(DEM_MAT)
dem  = np.array(_m['dem'], dtype=float)
lons = np.array(_m['longitude']).ravel(); lats = np.array(_m['latitude']).ravel()
dem[dem < 0] = np.nan
dlon = lons[1]-lons[0]; dlat0 = lats[0]-lats[1]

def at_nn(lo, la):
    i = int(np.argmin(np.abs(lats-la))); j = int(np.argmin(np.abs(lons-lo)))
    return dem[i, j]

def at_bi(lo, la):
    """【A3】双线性 DEM 插值。"""
    j = (lo-lons[0])/dlon; i = (lats[0]-la)/dlat0
    j0 = int(np.floor(j)); i0 = int(np.floor(i))
    if j0 < 0 or i0 < 0 or j0 > len(lons)-2 or i0 > len(lats)-2:
        return np.nan
    fj = j-j0; fi = i-i0
    return ((1-fj)*(1-fi)*dem[i0, j0] + fj*(1-fi)*dem[i0, j0+1]
            + (1-fj)*fi*dem[i0+1, j0] + fj*fi*dem[i0+1, j0+1])

def hav(lo1, la1, lo2, la2):
    r = radians
    a = sin((r(la2)-r(la1))/2)**2 + cos(r(la1))*cos(r(la2))*sin((r(lo2)-r(lo1))/2)**2
    return 6371000.0*2*atan2(sqrt(a), sqrt(1-a))

nodes = pd.read_csv(os.path.join(BASE, '预处理输出', 'Q1_节点_预处理.csv'))
q2    = pd.read_csv(os.path.join(BASE, 'Q2_主方案20架次_架次表.csv'))
uav   = pd.read_csv(os.path.join(BASE, '预处理输出', 'Q2_机型参数_预处理.csv'))
dfd   = pd.read_csv(os.path.join(BASE, 'Q2_主方案20架次_逐箱交付.csv'))

G01 = (float(nodes[nodes['编号']=='O01'].iloc[0]['经度']),
       float(nodes[nodes['编号']=='O01'].iloc[0]['纬度']),
       float(nodes[nodes['编号']=='O01'].iloc[0]['海拔(m)']) + 20.0)
O_ALT = 127.7
sa = nodes[nodes['编号'] != 'O01'].reset_index(drop=True)
pps = {r['机型编号']: dict(v=float(r['巡航速度(m/s)']), vu=float(r['最大爬升速度(m/s)']),
                           vd=float(r['最大下降速度(m/s)']), h0=float(r['基础交接时间(s)']),
                           h1=float(r['每箱增加交接时间(s)'])) for _, r in uav.iterrows()}
hop = {r['编号']: float(r['海拔(m)'])+30.0 for _, r in sa.iterrows()}; hop['O01'] = O_ALT
lon = {r['编号']: float(r['经度']) for _, r in sa.iterrows()}
lat = {r['编号']: float(r['纬度']) for _, r in sa.iterrows()}
lon['O01'] = G01[0]; lat['O01'] = G01[1]

def H_v2(a, b):
    ts = [0.0, 1.0]
    for axis, x1, x2 in [(lons, a[0], b[0]), (lats, a[1], b[1])]:
        bd = (axis[:-1]+axis[1:])/2.0
        t = (bd-x1)/(x2-x1)
        ts += t[(t > 0) & (t < 1)].tolist()
    ts = sorted(set(ts))
    sp = [0.0, 1.0] + [(u+v)/2 for u, v in zip(ts, ts[1:])]
    return max(at_nn(a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])) for t in sp) + 50.0

def H_center(a, b):
    ts = [0.0, 1.0]
    for axis, x1, x2 in [(lons, a[0], b[0]), (lats, a[1], b[1])]:
        t = (axis-x1)/(x2-x1)
        ts += t[(t > 0) & (t < 1)].tolist()
    return max(at_nn(a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])) for t in sorted(set(ts))) + 50.0

def build_traj(dt_step):
    """【A1】按相位以约 dt_step 秒步长采样；相位端点必取。"""
    rows = []
    for _, row in q2.iterrows():
        seq = row['路径'].split('→'); pp = pps[row['机型']]; fid = row['架次ID']
        bz = {}
        for _, c in dfd.iterrows():
            if c['架次ID'] == fid:
                bz.setdefault(c['服务区'], []).append(c['货箱序号'])
        t = float(row['起飞s']); prev = 'O01'
        for z in seq[1:-1] + ['O01']:
            pa, pb = (lon[prev], lat[prev]), (lon[z], lat[z])
            d = hav(pa[0], pa[1], pb[0], pb[1])
            Hc = H_center(pa, pb) if ('S007' in (prev, z)) else H_v2(pa, pb)
            h0, h1 = hop[prev], hop[z]
            tcl = max(Hc-h0, 0.0)/pp['vu']; tcr = d/pp['v']; tdn = max(Hc-h1, 0.0)/pp['vd']
            if prev == 'O01' and pp['vu'] == 3.0 and fid != 'T14':   # 经验项A
                tcl += 37.0
            for tag, t0, pA, pB, ha, hb, dtp in [
                    ('cl', t, prev, prev, h0, Hc, tcl),
                    ('cr', t+tcl, prev, z, Hc, Hc, tcr),
                    ('dn', t+tcl+tcr, z, z, Hc, h1, tdn)]:
                ns = max(1, int(dtp/dt_step))
                for k in range(ns+1):
                    u = k/ns
                    if tag == 'cr':
                        x = lon[pA] + (lon[pB]-lon[pA])*u
                        y = lat[pA] + (lat[pB]-lat[pA])*u
                    else:
                        x, y = lon[pA], lat[pA]
                    rows.append((t0+u*dtp, x, y, ha+(hb-ha)*u, fid, z))
            t += tcl+tcr+tdn
            if z != 'O01':
                hand = pp['h0'] + pp['h1']*len(bz.get(z, [])); ns = max(1, int(hand/dt_step))
                for k in range(ns+1):
                    rows.append((t+hand*k/ns, lon[z], lat[z], h1, fid, z))
                t += hand
            prev = z
    return pd.DataFrame(rows, columns=['t', 'lon', 'lat', 'h', '架次', '服务区'])

def judge(traj, pthr, hs_terr, h_add=0.0):
    """返回 (直连bool数组, Pr数组)。【A4】双向直连弱方向。
    hs_terr：加到地形上的“视线安全余量”（越大→遮挡越多→直连率越低）。
    h_add  ：加到飞行器高度的“安全超高”（越大→视线越高→直连率越高，论文口径）。"""
    cache = {}
    def gap(lo, la, h):
        key = (round(lo, 9), round(la, 9), round(h, 6), hs_terr)
        if key in cache:
            return cache[key]
        dd = hav(G01[0], G01[1], lo, la)
        ns = max(2, min(TERR_CAP, int(dd/TERR_STEP)+1))
        dep = -1e9
        for k in range(1, ns):
            tt = k/ns
            ht = at_bi(G01[0]+tt*(lo-G01[0]), G01[1]+tt*(la-G01[1]))
            hl = G01[2]+tt*(h-G01[2])
            if np.isfinite(ht):
                dep = max(dep, ht+hs_terr-hl)
        cache[key] = dep
        return dep
    pr = np.empty(len(traj))
    for i in range(len(traj)):
        lo, la, h = traj.iat[i, 1], traj.iat[i, 2], traj.iat[i, 3] + h_add
        d2 = hav(G01[0], G01[1], lo, la); d3 = sqrt(d2**2 + (G01[2]-h)**2)
        pl = FSPL_C + 20.0*log10(F_MHZ) + 20.0*log10(d3/1000.0) + (L_OBS if gap(lo, la, h) > 0.0 else 0.0)
        pr[i] = PR_TX_SUM - pl
    return pr >= pthr, pr

print('[1/2] 重建搜索口径轨迹（dt=%.0fs）...' % SAMPLING_DT)
traj5 = build_traj(SAMPLING_DT)
print('    dt=5s 轨迹点数 %d（对照 dt=1s 点数见下）' % len(traj5))

def zones_direct(traj, direct):
    """【A5】某服务区全部采样点直连 → 计入直连服务区。"""
    zs = [z for z in traj['服务区'].unique() if z != 'O01']
    n = 0
    for z in zs:
        m = traj['服务区'] == z
        if direct[m].all():
            n += 1
    return n, len(zs)

rows = []
# 门限敏感性（dt=5s, Hs=0）
print('[2/2] 门限敏感性（dt=5s, Hs=0）与 Hs 扫描（dt=5s, -90dBm）...')
for p in PTHRS:
    direct, pr = judge(traj5, p, 0.0)
    nd = int((~direct).sum())
    zok, ztot = zones_direct(traj5, direct)
    rows.append(dict(类别='接收门限', 参数='%.0f dBm' % p, 采样点数=len(traj5),
                     直连点数=int(direct.sum()), 需中继点数=nd,
                     直连率pct=round(100*direct.mean(), 2), 直连服务区数=zok, 服务区总数=ztot))
    print('   门限 %.0f dBm: 采样 %d, 直连 %d, 需中继 %d, 直连率 %.2f%%, 直连服务区 %d/%d'
          % (p, len(traj5), int(direct.sum()), nd, 100*direct.mean(), zok, ztot))
for h in HS_SCAN:
    dA, _ = judge(traj5, -90.0, hs_terr=h, h_add=0.0)
    zA, ztot = zones_direct(traj5, dA)
    rows.append(dict(类别='Hs扫描(地形余量)', 参数='Hs=%.0f m' % h, 采样点数=len(traj5),
                     直连点数=int(dA.sum()), 需中继点数=int((~dA).sum()),
                     直连率pct=round(100*dA.mean(), 2), 直连服务区数=zA, 服务区总数=ztot))
    dB, _ = judge(traj5, -90.0, hs_terr=0.0, h_add=h)
    zB, _ = zones_direct(traj5, dB)
    rows.append(dict(类别='Hs扫描(飞行超高)', 参数='Hs=%.0f m' % h, 采样点数=len(traj5),
                     直连点数=int(dB.sum()), 需中继点数=int((~dB).sum()),
                     直连率pct=round(100*dB.mean(), 2), 直连服务区数=zB, 服务区总数=ztot))
    print('   Hs=%.0f m: 地形余量→直连率 %.2f%% 需中继 %d；飞行超高→直连率 %.2f%% 需中继 %d'
          % (h, 100*dA.mean(), int((~dA).sum()), 100*dB.mean(), int((~dB).sum())))

# 附：dt=1s（论文主方案口径，Hs=0）作参照
traj1 = build_traj(1.0)
direct1, _ = judge(traj1, -90.0, 0.0)
zok1, ztot1 = zones_direct(traj1, direct1)
rows.append(dict(类别='采样步长参照', 参数='dt=1s, -90dBm, Hs=0', 采样点数=len(traj1),
                 直连点数=int(direct1.sum()), 需中继点数=int((~direct1).sum()),
                 直连率pct=round(100*direct1.mean(), 2), 直连服务区数=zok1, 服务区总数=ztot1))
print('   参照 dt=1s: 采样 %d, 需中继 %d, 直连率 %.2f%%' % (len(traj1), int((~direct1).sum()), 100*direct1.mean()))

res = pd.DataFrame(rows)
# 附论文声称值（仅作对照列，不参与计算）
claim = {('接收门限', '-90 dBm'): 2368, ('接收门限', '-95 dBm'): 806, ('接收门限', '-98 dBm'): 0}
res['论文声称需中继'] = [claim.get((r['类别'], r['参数']), '') for _, r in res.iterrows()]
res.to_csv(os.path.join(OUT, 'Q3_重建_门限敏感性.csv'), index=False, encoding='utf-8-sig')
print('-'*70)
print(res[['类别', '参数', '采样点数', '需中继点数', '直连率pct', '直连服务区数', '论文声称需中继']].to_string(index=False))
print('输出:', os.path.join(OUT, 'Q3_重建_门限敏感性.csv'))
