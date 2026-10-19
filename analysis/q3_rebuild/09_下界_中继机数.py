# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成。
"""
C：中继机数下界检验（可证命题，非声明）。

命题：本方案至少需要 2 架中继无人机。
证明思路（可计算验证）：
  R01M1 段1 与 R02M1 段1 的【在岗时窗重叠】（由规范需求集合推导），但其需求点云
  分别位于调度中心南侧与北侧，相距约 9 km。若候选悬停点网格（0.01°、离地 300 m、
  距 O01 ≤ 12 km）中不存在任何单点能同时覆盖两段的全部需求点，则两段必须由两个
  不同位置的悬停点服务；两个位置在时间上重叠 → 至少需要 2 架中继无人机同时在线。

输出：Q3_下界_中继机数.json
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from math import radians, log10

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_DEFAULT = os.path.normpath(os.path.join(_HERE, '..'))
BASE = os.environ.get('Q3_REPO', _REPO_DEFAULT)
OUT = os.environ.get('Q3_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs'))
DEM_MAT = os.path.join(BASE, '数据', '镇龙乡地理空间数据', '镇龙乡及周边地理数据', '数字高程模型数据（DEM）', '镇龙乡及周边30米DEM.mat')
GRID, R_MAX, HOVER_ABOVE, NS_PROF, HS = 0.01, 12000.0, 300.0, 30, 0.0
P_THR, C_FSPL = -90.0, 32.45

_m = loadmat(DEM_MAT)
dem = np.array(_m['dem'], float)
lons = np.array(_m['longitude']).ravel(); lats = np.array(_m['latitude']).ravel()
dem[dem < 0] = np.nan
dlon = lons[1]-lons[0]; dlat0 = lats[0]-lats[1]

def hav_m(lo1, la1, lo2, la2):
    a = np.sin((la2-la1)/2.0)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2.0)**2
    return 6371000.0*2*np.arctan2(np.sqrt(a), np.sqrt(1-a))

def terrain_nn(lo, la):
    ii = np.rint((lats[0]-la)/dlat0).astype(int); jj = np.rint((lo-lons[0])/dlon).astype(int)
    ok = (ii >= 0) & (ii < len(lats)) & (jj >= 0) & (jj < len(lons))
    o = np.full(lo.shape, np.nan)
    o[ok] = dem[np.clip(ii[ok], 0, len(lats)-1), np.clip(jj[ok], 0, len(lons)-1)]
    return o

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

_no = pd.read_csv(os.path.join(BASE, '预处理输出', 'Q1_节点_预处理.csv'))
_o = _no[_no['编号'] == 'O01'].iloc[0]
G01 = (float(_o['经度']), float(_o['纬度']), float(_o['海拔(m)']) + 20.0)

cand = []
for lo in np.arange(float(lons.min()), float(lons.max())+1e-9, GRID):
    for la in np.arange(float(lats.min()), float(lats.max())+1e-9, GRID):
        e = dem[int(np.argmin(np.abs(lats-la))), int(np.argmin(np.abs(lons-lo)))]
        if np.isfinite(e) and hav_m(radians(G01[0]), radians(G01[1]), radians(lo), radians(la)) < R_MAX:
            cand.append((lo, la, e+HOVER_ABOVE))
cand = np.array(cand)
print('候选悬停点:', len(cand))

pt = pd.read_csv(os.path.join(OUT, 'Q3_规范集合_覆盖核验逐点.csv'), encoding='utf-8-sig')

def cover_flags(P, C):
    d2 = hav_m(np.radians(P[:, 0, None]), np.radians(P[:, 1, None]),
               np.radians(C[None, :, 0]), np.radians(C[None, :, 1]))
    d3 = np.sqrt(d2**2 + (P[:, 2, None]-C[None, :, 2])**2)
    pra = 26.0 - (C_FSPL + 20*log10(2400.0) + 20*np.log10(d3/1000.0) + 10.0*blocked_mat(P, C))
    d2g = hav_m(np.radians(G01[0]), np.radians(G01[1]), np.radians(C[:, 0]), np.radians(C[:, 1]))
    d3g = np.sqrt(d2g**2 + (G01[2]-C[:, 2])**2)
    prg = 36.0 - (C_FSPL + 20*log10(2400.0) + 20*np.log10(d3g/1000.0)
                  + 10.0*blocked_mat(np.array([G01]), C)[0])
    return (pra >= P_THR) & (prg[None, :] >= P_THR)

s1 = pt[(pt['保障中继'] == 'R01M1') & (pt['时段'] == 1)]
s2 = pt[(pt['保障中继'] == 'R02M1') & (pt['时段'] == 1)]
print('R01M1段1 时窗 [%.1f, %.1f]，R02M1段1 时窗 [%.1f, %.1f]'
      % (s1['t'].min(), s1['t'].max(), s2['t'].min(), s2['t'].max()))
ov = min(s1['t'].max(), s2['t'].max()) - max(s1['t'].min(), s2['t'].min())
d_mid = hav_m(radians(s1['lon'].mean()), radians(s1['lat'].mean()),
              radians(s2['lon'].mean()), radians(s2['lat'].mean()))
print('两段时窗重叠 %.1f s，需求点云质心相距 %.0f m' % (ov, d_mid))

U = pd.concat([s1[['lon', 'lat', 'h']], s2[['lon', 'lat', 'h']]]).values
ok = cover_flags(U, cand)
best = int(ok.sum(axis=0).max())
n_full = int((ok.sum(axis=0) == len(U)).sum())
print('候选点中能同时覆盖两段全部 %d 个需求点的数量: %d（最佳单点覆盖 %d 个）'
      % (len(U), n_full, best))

# 并列检验：是否存在单点同时覆盖两段各自全部点
ok1 = cover_flags(s1[['lon', 'lat', 'h']].values, cand)
ok2 = cover_flags(s2[['lon', 'lat', 'h']].values, cand)
both = int(((ok1.sum(axis=0) == len(s1)) & (ok2.sum(axis=0) == len(s2))).sum())
print('可单独全覆盖段1 的候选点 %d 个；可单独全覆盖段1 的候选点 %d；两者兼具 %d 个'
      % (int((ok1.sum(axis=0) == len(s1)).sum()), int((ok2.sum(axis=0) == len(s2)).sum()), both))

res = dict(
    命题='本方案至少需要 2 架中继无人机',
    R01M1段1时窗=[float(s1['t'].min()), float(s1['t'].max())],
    R02M1段1时窗=[float(s2['t'].min()), float(s2['t'].max())],
    时窗重叠s=round(float(ov), 1),
    点云质心相距m=round(float(d_mid), 1),
    联合需求点数=int(len(U)),
    能单点全覆盖联合需求集的候选点数=int(n_full),
    最佳单点覆盖数=int(best),
    两段各自全覆盖且有交集候选点数=int(both),
    结论=('两段时窗重叠且不存在单点可同时覆盖两段全部需求 → 至少 2 架同时在线，命题成立'
          if (ov > 0 and both == 0) else '需人工复核'),
)
with open(os.path.join(OUT, 'Q3_下界_中继机数.json'), 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print('-'*80)
for k, v in res.items():
    print('  %-28s %s' % (k, v))
