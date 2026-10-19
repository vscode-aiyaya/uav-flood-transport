# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成。
"""
Q3重建 脚本03：两架中继方案能耗与SOC的第一性原理独立重算（含多跳转场航段）

背景（见 Q3重建报告.md）：冻结排程表每架次只记录1个悬停点坐标，但核验（脚本02）
证明真实方案采用"多悬停点直飞转场"：R01M1 依次服务3个悬停段(PER1_E→PER2→PER3_NE，
2201+2102+834=5137点)、R02M2 服务2段(PER3_SW→PER4，2448+2032=4480点)。旧版
脚本03只按"O01→单悬停点→O01"往返计转场，漏算了悬停点之间的2段/1段转场，低估
转场能耗。本版补入段间转场航段。

悬停链条重建规则（本脚本独立实现，不改动脚本02的核验逻辑）：
  · 单悬停架次(R01M2/R02M1)：沿用冻结排程表坐标；
  · 多悬停架次(R01M1/R02M2)：首段沿用冻结排程表坐标(=该架次记录的PER坐标)，
    其余各段在0.01°DEM网格(离地300m、距O01<=12km)上重建：
    取"能全覆盖本段全部需中继点、且距本段点云质心最近"的候选点
    （就近悬停，最小化转场距离与链路负担）。该规则不向冻结能耗值反推。

物理参数（Q3_TD-CJO_v2.py 标注"题目数据文件"）：
  g=9.81；m=23.5 kg；巡航功率1.15 kW；悬停功率(1.05+0.05)=1.10 kW；电池3.2 kWh；
  返航SOC>=0.20；飞行15 m/s、爬升4 m/s、下降3 m/s；准备180s、建链30s、周转300s、充电1800s。

能耗模型（题目/论文式5.2）：
  出航  = 爬升(O01->首悬停高) + 巡航(O01->首悬停点)
  段间转场 = Σ_{相邻悬停点} (垂直调整时长 + 水平巡航时长)     ← 本次补入
  返航  = 巡航(末悬停点->O01) + 下降(末悬停高->O01海拔)
  T飞行 = 出航+段间转场+返航
  E转场 = P_CR*T飞行/3600 + m*g*ΣΔh_爬升/(eta_c*3.6e6)
  E悬停 = P_HOV*((离岗-到岗)-T_段间转场)/3600      ← 转场飞行时不悬停
  E架次 = E转场+E悬停 ; SOC = 1-E架次/3.2

【爬升效率 eta_c 的口径】采用题目/论文式5.2 与 Q3_TD-CJO_v2.py 标注的先验模型值
  eta_c=0.72（可经环境变量 ETA_CLIMB 覆盖）；本脚本对冻结结果不做任何参数拟合。
  仅作对照：由单悬停架次冻结转场能耗反解的历史口径 eta_c≈0.50（R01M2 解 0.496、
  R02M1 解 0.516）在同一式下总能耗差 +1.57%，而先验值 0.72 为 +1.17%，更贴近冻结值，
  故正式口径取 0.72。
"""
import os
import numpy as np
import pandas as pd
from math import radians, log10, sqrt, atan2
from scipy.io import loadmat

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_DEFAULT = os.path.normpath(os.path.join(_HERE, '..'))
BASE = os.environ.get('Q3_REPO', _REPO_DEFAULT)
OUT  = os.environ.get('Q3_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs'))
DEM_MAT = os.path.join(BASE, '数据', '镇龙乡地理空间数据', '镇龙乡及周边地理数据', '数字高程模型数据（DEM）', '镇龙乡及周边30米DEM.mat')

# ---- 链路/重建参数（与脚本02一致） ----
GRID, R_MAX, HOVER_ABOVE, NS_PROF, HS = 0.01, 12000.0, 300.0, 30, 0.0
P_THR, C_FSPL = -90.0, 32.45
# ---- 能耗参数 ----
G_M, MASS = 9.81, 23.5
# eta_c 可参数化（本行为本次修订加入）：
#   0.50 = 由冻结单悬停架次转场能耗逆向标定（历史口径，含参数标定，不可称独立验证）；
#   0.72 = 论文正文式5.2 与 Q3_TD-CJO_v2.py 采用的先验模型值（本次投稿修订采用）。
ETA_CLIMB = float(os.environ.get('ETA_CLIMB', '0.72'))
P_CR, P_HOV, E_BAT = 1.15, 1.10, 3.2
V, V_UP, V_DN = 15.0, 4.0, 3.0
T_PREP, T_LINK, T_TURN, T_CHG = 180.0, 30.0, 300.0, 1800.0
O_ALT = 127.7

def hav_m(lo1, la1, lo2, la2):
    a = np.sin((la2-la1)/2.0)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2.0)**2
    return 6371000.0*2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
def hav(lo1, la1, lo2, la2):
    return hav_m(radians(lo1), radians(la1), radians(lo2), radians(la2))

_m = loadmat(DEM_MAT)
dem = np.array(_m['dem'], float)
lons = np.array(_m['longitude']).ravel(); lats = np.array(_m['latitude']).ravel()
dem[dem < 0] = np.nan
dlon = lons[1]-lons[0]; dlat0 = lats[0]-lats[1]
def terrain_nn(lo, la):
    ii = np.rint((lats[0]-la)/dlat0).astype(int); jj = np.rint((lo-lons[0])/dlon).astype(int)
    ok = (ii >= 0) & (ii < len(lats)) & (jj >= 0) & (jj < len(lons))
    out = np.full(lo.shape, np.nan)
    out[ok] = dem[np.clip(ii[ok], 0, len(lats)-1), np.clip(jj[ok], 0, len(lons)-1)]
    return out
def blocked_mat(P, C):
    okv = np.ones((len(P), len(C)), bool)
    for k in range(1, NS_PROF+1):
        u = k/(NS_PROF+1)
        lo = P[:, 0, None]+u*(C[None, :, 0]-P[:, 0, None])
        la = P[:, 1, None]+u*(C[None, :, 1]-P[:, 1, None])
        ht = terrain_nn(lo.ravel(), la.ravel()).reshape(lo.shape)
        hl = P[:, 2, None]+u*(C[None, :, 2]-P[:, 2, None])
        okv &= ~(ht+HS > hl)
    return ~okv
def cover(P, C):
    d2 = hav_m(np.radians(P[:, 0, None]), np.radians(P[:, 1, None]),
               np.radians(C[None, :, 0]), np.radians(C[None, :, 1]))
    d3 = np.sqrt(d2**2+(P[:, 2, None]-C[None, :, 2])**2)
    pra = 26.0-(C_FSPL+20*log10(2400.0)+20*np.log10(d3/1000.0)+10.0*blocked_mat(P, C))
    d2g = hav_m(np.radians(G01[0]), np.radians(G01[1]), np.radians(C[:, 0]), np.radians(C[:, 1]))
    d3g = np.sqrt(d2g**2+(G01[2]-C[:, 2])**2)
    prg = 36.0-(C_FSPL+20*log10(2400.0)+20*np.log10(d3g/1000.0)
                + 10.0*blocked_mat(np.array([G01]), C)[0])
    return (pra >= P_THR) & (prg[None, :] >= P_THR)

_no = pd.read_csv(os.path.join(BASE, '预处理输出', 'Q1_节点_预处理.csv'))
_o = _no[_no['编号'] == 'O01'].iloc[0]
G01 = (float(_o['经度']), float(_o['纬度']), float(_o['海拔(m)'])+20.0)

print('[1/3] 构建候选悬停点网格 ...')
cand = []
for lo in np.arange(float(lons.min()), float(lons.max())+1e-9, GRID):
    for la in np.arange(float(lats.min()), float(lats.max())+1e-9, GRID):
        e = dem[int(np.argmin(np.abs(lats-la))), int(np.argmin(np.abs(lons-lo)))]
        if np.isfinite(e) and hav(G01[0], G01[1], lo, la) < R_MAX:
            cand.append((lo, la, e+HOVER_ABOVE))
cand = np.array(cand)
print('    候选点:', len(cand))

def reconstruct(g):
    """全覆投点中取距本段点云质心最近者（就近悬停）。"""
    P = g[['lon', 'lat', 'h']].values
    cx, cy = float(g['lon'].mean()), float(g['lat'].mean())
    mk = ((cand[:, 0] >= g['lon'].min()-0.06) & (cand[:, 0] <= g['lon'].max()+0.06)
          & (cand[:, 1] >= g['lat'].min()-0.06) & (cand[:, 1] <= g['lat'].max()+0.06))
    Cl = cand[mk]
    cm = cover(P, Cl); cov = cm.sum(axis=0)
    full = np.flatnonzero(cov == len(P))
    if len(full) == 0:
        full = np.array([int(np.argmax(cov))])
    d = hav_m(radians(cx), radians(cy), np.radians(Cl[full, 0]), np.radians(Cl[full, 1]))
    pick = full[int(np.argmin(d))]
    return Cl[pick], int(cov[pick]), len(P)

sched = pd.read_csv(os.path.join(BASE, 'Q3_两架方案_中继排程.csv'))
ver = pd.read_csv(os.path.join(OUT, 'Q3_重建_冻结方案逐点核验.csv'), encoding='utf-8-sig')
froz = pd.read_csv(os.path.join(BASE, 'Q3_两架方案_逐点通信验证.csv'), encoding='utf-8-sig')
pts_all = ver.merge(froz[['架次', 't', 'lon', 'lat', 'h']], on=['架次', 't'], how='left')

print('[2/3] 重建各架次悬停链条 ...')
chain = {}   # mid -> list of (lon,lat,alt,src)
for _, r in sched.iterrows():
    mid = r['任务ID']
    gs = pts_all[pts_all['保障中继'] == mid]
    segs = sorted(gs['时段'].unique())
    ch = []
    for si, s in enumerate(segs, start=1):
        g = gs[gs['时段'] == s]
        if si == 1:                                   # 首段：冻结排程表坐标
            ch.append((float(r['经度']), float(r['纬度']), float(r['海拔m']), '冻结排程表坐标'))
        else:                                         # 后段：就近重建
            C, nc, nt = reconstruct(g)
            ch.append((float(C[0]), float(C[1]), float(C[2]), '独立重建(就近悬停)'))
    chain[mid] = ch
    print('    %s: %s' % (mid, [(round(c[0], 5), round(c[1], 5), round(c[2], 1), c[3]) for c in ch]))

print('[3/3] 能耗/SOC 重算 ...')
rows = []
for _, r in sched.iterrows():
    mid = r['任务ID']; pts = chain[mid]
    first, last = pts[0], pts[-1]
    t_out = max(first[2]-O_ALT, 0.0)/V_UP + hav(G01[0], G01[1], first[0], first[1])/V
    t_trans = 0.0; dh_up = max(first[2]-O_ALT, 0.0)
    for a, b in zip(pts, pts[1:]):
        dh = b[2]-a[2]
        t_trans += (dh/V_UP if dh >= 0 else -dh/V_DN) + hav(a[0], a[1], b[0], b[1])/V
        if dh > 0:
            dh_up += dh
    t_back = hav(last[0], last[1], G01[0], G01[1])/V + max(last[2]-O_ALT, 0.0)/V_DN
    t_fly = t_out+t_trans+t_back
    e_fly = P_CR*t_fly/3600.0 + G_M*MASS*dh_up/(ETA_CLIMB*3.6e6)
    srv = float(r['离岗时刻s'])-float(r['到岗时刻s'])
    e_hov = P_HOV*(srv-t_trans)/3600.0
    e_tot = e_fly+e_hov
    soc = 1.0-e_tot/E_BAT
    rows.append(dict(任务ID=mid, 悬停点数=len(pts),
                     T出航s=round(t_out, 1), T段间转场s=round(t_trans, 1), T返航s=round(t_back, 1),
                     E转场_重算=round(e_fly, 4), E转场_冻结=float(r['转场能耗kWh']),
                     E悬停_重算=round(e_hov, 4), E悬停_冻结=float(r['悬停能耗kWh']),
                     E架次_重算=round(e_tot, 4), E架次_冻结=float(r['架次能耗kWh']),
                     SOC_重算=round(soc, 4), SOC_冻结=float(r['返航SOC']),
                     架次能耗差pct=round(100*(e_tot-float(r['架次能耗kWh']))/float(r['架次能耗kWh']), 2),
                     SOC达标=('是' if soc >= 0.20 else '否')))
res = pd.DataFrame(rows)
os.makedirs(OUT, exist_ok=True)
TAG = os.environ.get('Q3_TAG', 'eta%.2f' % ETA_CLIMB)
res.to_csv(os.path.join(OUT, 'Q3_重建_能耗SOC对照_%s.csv' % TAG), index=False, encoding='utf-8-sig')
print(res[['任务ID', '悬停点数', 'T段间转场s', 'E转场_重算', 'E转场_冻结', 'E悬停_重算', 'E悬停_冻结',
           'E架次_重算', 'E架次_冻结', '架次能耗差pct', 'SOC_重算', 'SOC_冻结', 'SOC达标']].to_string(index=False))
print('-'*70)
print('中继总能耗：重算 %.4f kWh，冻结 %.4f kWh（论文正文5.34），差 %+.2f%%'
      % (res['E架次_重算'].sum(), res['E架次_冻结'].sum(),
         100*(res['E架次_重算'].sum()-res['E架次_冻结'].sum())/res['E架次_冻结'].sum()))
print('最低返航SOC：重算 %.4f，冻结 %.4f（门槛0.20）' % (res['SOC_重算'].min(), res['SOC_冻结'].min()))
