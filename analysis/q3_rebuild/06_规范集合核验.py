# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成。
"""
A3：在【规范需求集合】上对【固定主方案】做只读覆盖核验。

与脚本02的区别与关系：
  · 脚本02在冻结的 11683 点上核验，并对冻结文件缺失的 2 个悬停点做一次重建；
  · 本脚本不改动方案：计划直接取自脚本02输出的《核验汇总》（4个冻结悬停坐标 +
    2个重建补齐坐标，共 7 个悬停段），随后**只读**核验脚本01生成的规范需求集合
    （11713 点，含冻结集合中没有的 30 个 T13/T14/T18 边缘点）。
  · 核验器不搜索、不重建，只按计划给定的坐标与服务时窗逐点判定双向链路。

点—任务归属规则（对规范集合中的每一点）：
  1) 该点所属架次在冻结文件中已登记的中继任务集合 R(架次)；
  2) 在 R(架次) 内取“服务时窗包含该点时刻 t”的任务；
  3) 若多个，取窗口区间长度最小者；若无窗口包含，取窗口中心距 t 最近者（并标注“窗近似”）。

链路参数与脚本02完全一致（论文式5.1/表C.2）：TR<->REL 弱向 26 dB、REL<->G01 弱向
36 dB、FSPL 常数 32.45、门限 -90 dBm、最近像元 DEM、视线剖面 30 个内部采样、Hs=0。

输出：
  Q3_规范集合_覆盖核验逐点.csv
  Q3_规范集合_覆盖核验汇总.csv
  Q3_规范集合_新增边缘点.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from math import radians, log10

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_DEFAULT = os.path.normpath(os.path.join(_HERE, '..'))
BASE = os.environ.get('Q3_REPO', _REPO_DEFAULT)
OUT = os.environ.get('Q3_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs'))
DEM_MAT = os.path.join(BASE, '数据', '镇龙乡地理空间数据', '镇龙乡及周边地理数据', '数字高程模型数据（DEM）', '镇龙乡及周边30米DEM.mat')

NS_PROF, HS = 30, 0.0
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

_no = pd.read_csv(os.path.join(BASE, '预处理输出', 'Q1_节点_预处理.csv'))
_o = _no[_no['编号'] == 'O01'].iloc[0]
G01 = (float(_o['经度']), float(_o['纬度']), float(_o['海拔(m)']) + 20.0)

def cover_one(p, c):
    """单点双向链路判定，返回 (是否覆盖, TR中继弱向Pr, 中继G01弱向Pr)。"""
    P = np.array([[p[0], p[1], p[2]]]); C = np.array([[c[0], c[1], c[2]]])
    d2 = hav_m(np.radians(P[:, 0, None]), np.radians(P[:, 1, None]),
               np.radians(C[None, :, 0]), np.radians(C[None, :, 1]))
    d3 = np.sqrt(d2**2 + (P[:, 2, None]-C[None, :, 2])**2)
    pra = 26.0 - (C_FSPL + 20*log10(2400.0) + 20*np.log10(d3/1000.0) + 10.0*blocked_mat(P, C))
    d2g = hav_m(np.radians(G01[0]), np.radians(G01[1]), np.radians(C[:, 0]), np.radians(C[:, 1]))
    d3g = np.sqrt(d2g**2 + (G01[2]-C[:, 2])**2)
    prg = 36.0 - (C_FSPL + 20*log10(2400.0) + 20*np.log10(d3g/1000.0)
                  + 10.0*blocked_mat(np.array([G01]), C)[0])
    ok = bool((pra[0, 0] >= P_THR) and (prg[0] >= P_THR))
    return ok, float(pra[0, 0]), float(prg[0])

# ---- 固定计划：7 个悬停段（冻结坐标 + 02 重建补齐坐标），只读 ----
plan = pd.read_csv(os.path.join(OUT, 'Q3_重建_冻结方案核验汇总.csv'), encoding='utf-8-sig')
PLAN = {int(r['时段']) if False else (r['任务ID'], int(r['时段'])):
        dict(lon=float(r['悬停经度']), lat=float(r['悬停纬度']), alt=float(r['悬停海拔m']),
             t0=float(r['t起']), t1=float(r['t止']), src=r['悬停点来源'])
        for _, r in plan.iterrows()}
print('计划悬停段数: %d' % len(PLAN))
for k, v in PLAN.items():
    print('  %s 段%d  t=[%.1f, %.1f]  src=%s' % (k[0], k[1], v['t0'], v['t1'], v['src']))

# ---- 冻结文件：架次->中继任务集合（用于给规范集合新增点定任务） ----
froz = pd.read_csv(os.path.join(BASE, 'Q3_两架方案_逐点通信验证.csv'), encoding='utf-8-sig')
sa2rel = froz.groupby('架次')['保障中继'].apply(lambda s: sorted(set(s))).to_dict()
# 冻结时间轴（按架次排序），用于容差最近邻判定“该规范点是否已在冻结集合中”
froz_ts = {sa: np.sort(g['t'].values.astype(float)) for sa, g in froz.groupby('架次')}
TOL = 0.5   # 秒；两套时间网格的节点不一致（见报告 §1），此处用容差而非精确相等

def is_in_frozen(sa, t):
    a = froz_ts.get(sa)
    if a is None or len(a) == 0:
        return False
    i = int(np.searchsorted(a, t))
    for j in (i-1, i):
        if 0 <= j < len(a) and abs(a[j]-t) <= TOL:
            return True
    return False

# ---- 规范需求集合（11713，脚本01） ----
need = pd.read_csv(os.path.join(OUT, 'Q3_重建_需中继点.csv'), encoding='utf-8-sig')
need['in_frozen'] = [is_in_frozen(a, float(t)) for a, t in zip(need['架次'], need['t'])]
print('规范需求集合 %d 点；其中不在冻结集合中的新增点 %d 个'
      % (len(need), int((~need['in_frozen']).sum())))

def pick_task(sa, t):
    """返回 (任务键, 归属方式)。"""
    cands = [k for k in PLAN if k[0] in sa2rel.get(sa, [])]
    if not cands:
        cands = list(PLAN.keys())
    inside = [k for k in cands if PLAN[k]['t0'] <= t <= PLAN[k]['t1']]
    if inside:
        k = min(inside, key=lambda k: PLAN[k]['t1'] - PLAN[k]['t0'])
        return k, '时窗'
    k = min(cands, key=lambda k: abs((PLAN[k]['t0'] + PLAN[k]['t1'])/2.0 - t))
    return k, '窗近似'

rows = []
for _, p in need.iterrows():
    sa, t = p['架次'], float(p['t'])
    k, how = pick_task(sa, t)
    c = PLAN[k]
    ok, pr_tr, pr_g01 = cover_one((float(p['lon']), float(p['lat']), float(p['h'])),
                                  (c['lon'], c['lat'], c['alt']))
    rows.append(dict(架次=sa, t=round(t, 1), lon=round(float(p['lon']), 6),
                     lat=round(float(p['lat']), 6), h=round(float(p['h']), 2),
                     保障中继=k[0], 时段=k[1], 悬停点来源=c['src'], 归属方式=how,
                     在冻结集合=bool(p['in_frozen']),
                     TR中继弱向Pr=round(pr_tr, 2), 中继G01弱向Pr=round(pr_g01, 2),
                     核验通过=int(ok)))
res = pd.DataFrame(rows)
res = res.sort_values(['架次', 't']).reset_index(drop=True)
res.drop(columns=['在冻结集合']).to_csv(os.path.join(OUT, 'Q3_规范集合_覆盖核验逐点.csv'),
                                        index=False, encoding='utf-8-sig')

# ---- 汇总 ----
summary = res.groupby(['保障中继', '时段', '悬停点来源', '归属方式']).agg(
    点数=('核验通过', 'size'), 通过=('核验通过', 'sum'),
    TR最弱Pr=('TR中继弱向Pr', 'min'), G01最弱Pr=('中继G01弱向Pr', 'min')).reset_index()
summary['通过率'] = (summary['通过'] / summary['点数']).round(6)
summary.to_csv(os.path.join(OUT, 'Q3_规范集合_覆盖核验汇总.csv'), index=False, encoding='utf-8-sig')

extra = res[~res['在冻结集合']].copy()
extra.to_csv(os.path.join(OUT, 'Q3_规范集合_新增边缘点.csv'), index=False, encoding='utf-8-sig')

print('-'*70)
print('规范集合核验：总点 %d，通过 %d，覆盖率 %.4f%%'
      % (len(res), int(res['核验通过'].sum()), 100.0*res['核验通过'].mean()))
print('  其中冻结集合内 %d 点，通过 %d' % ((res['在冻结集合']).sum(),
                                        int(res.loc[res['在冻结集合'], '核验通过'].sum())))
print('  其中新增边缘点 %d 点，通过 %d' % ((~res['在冻结集合']).sum(),
                                        int(res.loc[~res['在冻结集合'], '核验通过'].sum())))
print('全局最弱：TR中继弱向 %.2f dBm，中继G01弱向 %.2f dBm'
      % (res['TR中继弱向Pr'].min(), res['中继G01弱向Pr'].min()))
print('归属方式分布：', res['归属方式'].value_counts().to_dict())
if len(extra):
    print('\n新增边缘点明细：')
    print(extra[['架次', 't', 'lon', 'lat', 'h', '保障中继', '时段', '归属方式',
                 'TR中继弱向Pr', '中继G01弱向Pr', '核验通过']].to_string(index=False))
bad = res[res['核验通过'] == 0]
print('\n未通过点数：', len(bad))
if len(bad):
    print(bad[['架次', 't', '保障中继', '时段', 'TR中继弱向Pr', '中继G01弱向Pr']].to_string(index=False))
