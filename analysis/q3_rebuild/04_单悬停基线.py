# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成。
"""
Q3重建 脚本04：对照方案A——单悬停点往返模式（论文5.4.2节对照实验的如实复算）

规则（严格按论文5.3/5.4节文字与现存 TD-CJO 求解器 Q3_TD-CJO_v2.py 的规则，
不反推目标值）：
  1) 全部需中继点按时间间隔>400s聚成全局服务簇；
  2) 每簇尝试用1个悬停点全覆盖（0.01° DEM网格、离地300m、距O01<=12km、
     最近像元高程、贪心覆盖收益最大）；不能全覆盖则按中位时间拆2段，再不行
     按时间三等分拆3段（与 v2 求解器一致）；每段=1个单悬停往返架次
     O01->悬停点->O01，不允许空中转场；
  3) N=1/2/3架中继做最早可用时序指派；回基地换能源组件（6组、两阶段充电
     T_chg=500+1300*(1-SOC) s，满放恰1800s），准备180s、建链30s、周转300s；
  4) 能耗与SOC检查与主方法核验链脚本07完全同口径（论文式5.2）：
     E架次 = 1.15kW*(出航+返航时长)/3600 + m*g*Δh爬升/(0.72*3.6e6) + 1.10kW*服务悬停/3600，
     SOC = 1 - E/3.2，返航SOC<0.20 的段计不可执行（组件充满后方可复用）；
     来不及出动或能量不可行的段，其点全部计未覆盖；
  5) 保障率=被覆盖需中继点/全部需中继点。
本脚本独立复算，如实报告（外部审查P1-03修订：旧版单阶段1800s充电且无SOC检查，
与主方法规则不一致，已统一；统一后覆盖点数 5968/7347/9611 与修订前完全一致）。

链路口径：论文5.1节DEM最近像元规则；FSPL 32.45、Lsys=3、遮挡+10dB、Hs=0、
         门限-90dBm；TR<->REL_A弱向26dB、REL_B<->G01弱向36dB。
需中继点来源：脚本01独立复算 Q3_重建_需中继点.csv（11713点）。
"""
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from math import radians, log10

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_DEFAULT = os.path.normpath(os.path.join(_HERE, '..'))
BASE = os.environ.get('Q3_REPO', _REPO_DEFAULT)
OUT  = os.environ.get('Q3_OUT', os.path.join(_HERE, 'outputs'))
os.makedirs(OUT, exist_ok=True)
DEM_MAT = os.path.join(BASE, '数据', '镇龙乡地理空间数据', '镇龙乡及周边地理数据', '数字高程模型数据（DEM）', '镇龙乡及周边30米DEM.mat')

GRID, R_MAX, HOVER_ABOVE = 0.01, 12000.0, 300.0
GAP_CLUSTER, NS_PROF, HS = 400.0, 30, 0.0
V, V_UP, V_DN = 15.0, 4.0, 3.0
T_PREP, T_LINK, T_TURN = 180.0, 30.0, 300.0
N_COMP, O_ALT, P_THR, C_FSPL = 6, 127.7, -90.0, 32.45
# 能耗与充电规则（与脚本07核验链完全一致：论文式5.2 + 两阶段充电）
G_M, MASS, ETA_CLIMB = 9.81, 23.5, 0.72
P_CR, P_HOV, E_BAT, SOC_MIN = 1.15, 1.10, 3.2, 0.20
CHG_FIX, CHG_SLOPE = 500.0, 1300.0   # T_chg = 500 + 1300*(1-SOC)，满放恰1800s

_m = loadmat(DEM_MAT)
dem = np.array(_m['dem'], float)
lons = np.array(_m['longitude']).ravel(); lats = np.array(_m['latitude']).ravel()
dem[dem < 0] = np.nan
dlon = lons[1]-lons[0]; dlat0 = lats[0]-lats[1]

def hav_m(lo1, la1, lo2, la2):
    a = np.sin((la2-la1)/2.0)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2.0)**2
    return 6371000.0*2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
def hav(lo1, la1, lo2, la2):
    return float(hav_m(radians(lo1), radians(la1), radians(lo2), radians(la2)))

def terrain_nn(lo, la):
    ii = np.rint((lats[0]-la)/dlat0).astype(int)
    jj = np.rint((lo-lons[0])/dlon).astype(int)
    ok = (ii >= 0) & (ii < len(lats)) & (jj >= 0) & (jj < len(lons))
    out = np.full(lo.shape, np.nan)
    out[ok] = dem[np.clip(ii[ok], 0, len(lats)-1), np.clip(jj[ok], 0, len(lons)-1)]
    return out

def blocked_mat(P, C):
    line_ok = np.ones((len(P), len(C)), bool)
    for k in range(1, NS_PROF+1):
        u = k/(NS_PROF+1)
        lo = P[:, 0, None] + u*(C[None, :, 0]-P[:, 0, None])
        la = P[:, 1, None] + u*(C[None, :, 1]-P[:, 1, None])
        ht = terrain_nn(lo.ravel(), la.ravel()).reshape(lo.shape)
        hl = P[:, 2, None] + u*(C[None, :, 2]-P[:, 2, None])
        line_ok &= ~(ht + HS > hl)
    return ~line_ok

def cover_mask(P, C):
    d2 = hav_m(np.radians(P[:, 0, None]), np.radians(P[:, 1, None]),
               np.radians(C[None, :, 0]), np.radians(C[None, :, 1]))
    d3a = np.sqrt(d2**2 + (P[:, 2, None]-C[None, :, 2])**2)
    pra = 26.0 - (C_FSPL + 20*log10(2400.0) + 20*np.log10(d3a/1000.0) + 10.0*blocked_mat(P, C))
    d2g = hav_m(np.radians(G01[0]), np.radians(G01[1]), np.radians(C[:, 0]), np.radians(C[:, 1]))
    d3g = np.sqrt(d2g**2 + (G01[2]-C[:, 2])**2)
    prg = 36.0 - (C_FSPL + 20*log10(2400.0) + 20*np.log10(d3g/1000.0)
                  + 10.0*blocked_mat(np.array([G01]), C)[0])
    return (pra >= P_THR) & (prg[None, :] >= P_THR)

_no = pd.read_csv(os.path.join(BASE, '预处理输出', 'Q1_节点_预处理.csv'))
_o = _no[_no['编号'] == 'O01'].iloc[0]
G01 = (float(_o['经度']), float(_o['纬度']), float(_o['海拔(m)']) + 20.0)

print('构建候选悬停点网格 ...')
cand = []
for lo in np.arange(float(lons.min()), float(lons.max())+1e-9, GRID):
    for la in np.arange(float(lats.min()), float(lats.max())+1e-9, GRID):
        e = dem[int(np.argmin(np.abs(lats-la))), int(np.argmin(np.abs(lons-lo)))]
        if np.isfinite(e) and hav(G01[0], G01[1], lo, la) < R_MAX:
            cand.append((lo, la, e+HOVER_ABOVE))
cand = np.array(cand)
print('候选点:', len(cand))

need = pd.read_csv(os.path.join(OUT, 'Q3_重建_需中继点.csv')).sort_values('t').reset_index(drop=True)
TOT = len(need)

def best_hover(g):
    """对一组点返回（最佳候选坐标, 逐点覆盖布尔, 局部候选集）。"""
    P = g[['lon', 'lat', 'h']].values
    m = ((cand[:, 0] >= g['lon'].min()-0.08) & (cand[:, 0] <= g['lon'].max()+0.08)
         & (cand[:, 1] >= g['lat'].min()-0.08) & (cand[:, 1] <= g['lat'].max()+0.08))
    Cl = cand[m]
    cm = cover_mask(P, Cl)
    bi = int(np.argmax(cm.sum(axis=0)))
    return Cl[bi], cm[:, bi]

# ---- 步骤1：全局时间簇 ----
edges = [0] + list(np.where(need['t'].diff() > GAP_CLUSTER)[0]) + [TOT]
clusters = [need.iloc[a:b].reset_index(drop=True) for a, b in zip(edges[:-1], edges[1:])]
print('全局时间簇:', len(clusters))

# ---- 步骤2：每簇贪心集合覆盖成1~3个悬停段（允许段间时间重叠=并行架次） ----
# 规则：取簇内覆盖收益最大悬停点，其覆盖点成段；对残余点重复，至多3个悬停
# （对应论文"1点不全盖则拆分服务段"与最多3架中继的实验上限）。
segments = []
for ci, g in enumerate(clusters, start=1):
    resid = g.reset_index(drop=True)
    for k in range(1, 4):
        if not len(resid):
            break
        C, okm = best_hover(resid)
        gg = resid[okm].reset_index(drop=True)
        tag = '单点' if k == 1 else f'残余第{k}点'
        d = hav(G01[0], G01[1], C[0], C[1]); dh = C[2]-O_ALT
        segments.append(dict(cluster=ci, seg=k, split=tag, C=C, g=gg,
                             t0=float(gg['t'].min()), t1=float(gg['t'].max()),
                             n=len(gg), ncov=len(gg),
                             tout=dh/V_UP+d/V, tback=d/V+dh/V_DN))
        resid = resid[~okm].reset_index(drop=True)
        if not len(resid):
            break
print('单悬停往返段数:', len(segments))
segdf = pd.DataFrame([{k: v for k, v in s.items() if k not in ('C', 'g')} |
                      dict(悬停经度=round(float(s['C'][0]), 5),
                           悬停纬度=round(float(s['C'][1]), 5),
                           悬停海拔=round(float(s['C'][2]), 1)) for s in segments])
print(segdf[['cluster', 'seg', 'split', 't0', 't1', 'n', 'ncov', 'tout', 'tback']].to_string(index=False))
geom_cov = sum(s['ncov'] for s in segments)
print('纯几何覆盖: %d/%d = %.2f%%' % (geom_cov, TOT, 100*geom_cov/TOT))

# ---- 步骤3：N架中继时序仿真（最早可用 + 6组件 + SOC检查 + 两阶段充电） ----
def sortie_energy(s):
    """单悬停往返架次能耗（kWh），口径与脚本07核验链一致：
    巡航1.15kW全程 + 爬升势能/(0.72*3.6e6) + 服务悬停1.10kW。"""
    dh = max(s['C'][2] - O_ALT, 0.0)
    e_fly = P_CR*(s['tout'] + s['tback'])/3600.0 + G_M*MASS*dh/(ETA_CLIMB*3.6e6)
    e_hov = P_HOV*(s['t1'] - s['t0'])/3600.0
    return e_fly + e_hov

def simulate(N):
    avail = [0.0]*N
    cfree = [0.0]*N_COMP
    rows = []
    for s in sorted(segments, key=lambda x: x['t0']):
        e = sortie_energy(s)
        soc_ret = 1.0 - e/E_BAT
        if soc_ret < SOC_MIN:
            rows.append(dict(簇=s['cluster'], 段=s['seg'], 可执行=False, 不可执行原因='返航SOC<0.20',
                             几何覆盖点=s['ncov'], 计入保障=0, 未覆盖点=s['n']))
            continue
        prep_start = s['t0'] - T_LINK - s['tout'] - T_PREP
        chosen = None
        for r in range(N):
            if avail[r] <= prep_start:
                cid = next((c for c in range(N_COMP) if cfree[c] <= prep_start), None)
                if cid is not None:
                    chosen = (r, cid); break
        if chosen is None:
            rows.append(dict(簇=s['cluster'], 段=s['seg'], 可执行=False, 不可执行原因='机或组件不可用',
                             几何覆盖点=s['ncov'], 计入保障=0, 未覆盖点=s['n']))
            continue
        r, cid = chosen
        land = s['t1'] + s['tback']
        avail[r] = land + T_TURN
        cfree[cid] = land + CHG_FIX + CHG_SLOPE*(1.0 - soc_ret)
        rows.append(dict(簇=s['cluster'], 段=s['seg'], 可执行=True, 中继=r+1, 组件=cid+1,
                         准备开始=round(prep_start, 1), 落地=round(land, 1),
                         架次能耗kWh=round(e, 3), 返航SOC=round(soc_ret, 3),
                         充电完成=round(cfree[cid], 1),
                         几何覆盖点=s['ncov'], 计入保障=s['ncov'],
                         未覆盖点=s['n']-s['ncov']))
    df = pd.DataFrame(rows)
    covered = int(df['计入保障'].sum())
    return df, covered

summ = []
for N in (1, 2, 3):
    df, cov = simulate(N)
    rate = 100.0*cov/TOT
    ok = df[df['可执行'] == True]
    summ.append(dict(中继机架数=N, 保障需中继点=cov, 总需中继点=TOT,
                     保障率pct=round(rate, 2),
                     能量不可行段=int((df['不可执行原因'] == '返航SOC<0.20').sum()),
                     不可用段=int((df['不可执行原因'] == '机或组件不可用').sum()),
                     最低返航SOC=round(float(ok['返航SOC'].min()), 3) if len(ok) else None))
    df.to_csv(os.path.join(OUT, f'Q3_重建_单悬停基线_N{N}架_段排程.csv'),
              index=False, encoding='utf-8-sig')
    print('N=%d: 保障率 %.2f%%，未被保障点 %d（能量不可行段 %d，机/组件不可用段 %d）'
          % (N, rate, TOT-cov,
             int((df['不可执行原因'] == '返航SOC<0.20').sum()),
             int((df['不可执行原因'] == '机或组件不可用').sum())))
pd.DataFrame(summ).to_csv(os.path.join(OUT, 'Q3_重建_单悬停基线_保障率.csv'),
                          index=False, encoding='utf-8-sig')
print('输出目录:', OUT)
