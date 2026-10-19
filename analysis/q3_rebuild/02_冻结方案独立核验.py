# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成。
"""
Q3重建 脚本02：冻结《两架方案》的独立链路核验与缺失悬停点重建

背景（经核查确认）：
  冻结排程表 Q3_两架方案_中继排程.csv 每架次仅记录1个悬停点坐标（共4个），
  而论文5.3节记载4架次共使用7个悬停点（R01M1服务3个时段、R02M2服务2个）。
  对冻结11683点按时序间隔>400s分簇，恰得7个服务时段；用排程表坐标核验：
    R01M1 坐标可覆盖其时段1、2（2201+2102点），时段3不覆盖（仅68/834）
    R01M2 坐标覆盖其全部508点
    R02M1 坐标覆盖其全部1558点
    R02M2 坐标覆盖时段1（2448点），时段2仅1578/2032
  即缺失2个转场悬停点（R01M1时段3、R02M2时段2），本脚本对其按论文5.3.2节
  规则在0.01°DEM网格（离地300m、距O01<=12km）上独立重建（贪心覆盖最大点），
  再做逐点双跳双向链路核验。重建点已在输出与报告中显式标注，不与冻结实测点混淆。

链路参数（论文式5.1、表C.2）：
  TR(20dBm,3dBi)<->REL_A(20dBm,6dBi) 双向弱向 26-PL
  REL_B(19dBm,8dBi)<->G01(27dBm,12dBi) 双向弱向 36-PL
  FSPL常数32.45、Lsys=3dB、遮挡+10dB、门限-90dBm、Hs=0；
  DEM最近像元（论文5.1节"与问题一、二最近像元规则一致"），视线剖面30个内部采样。

输出：
  outputs\\Q3_重建_冻结方案逐点核验.csv
  outputs\\Q3_重建_冻结方案核验汇总.csv
  outputs\\Q3_重建_缺失悬停点.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from math import radians, log10

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_DEFAULT = os.path.normpath(os.path.join(_HERE, '..'))
BASE = os.environ.get('Q3_REPO', _REPO_DEFAULT)
OUT  = os.environ.get('Q3_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs'))
os.makedirs(OUT, exist_ok=True)
DEM_MAT = os.path.join(BASE, '数据', '镇龙乡地理空间数据', '镇龙乡及周边地理数据', '数字高程模型数据（DEM）', '镇龙乡及周边30米DEM.mat')
GRID, R_MAX, HOVER_ABOVE, GAP_CLUSTER, NS_PROF, HS = 0.01, 12000.0, 300.0, 400.0, 30, 0.0
P_THR, C_FSPL = -90.0, 32.45

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

def cover(P, C, pr_only=False):
    """返回 (cover_bool(n,m), 弱向Pr_TR中继(n,m), Pr_中继G01(m,))。"""
    d2 = hav_m(np.radians(P[:, 0, None]), np.radians(P[:, 1, None]),
               np.radians(C[None, :, 0]), np.radians(C[None, :, 1]))
    d3 = np.sqrt(d2**2 + (P[:, 2, None]-C[None, :, 2])**2)
    pra = 26.0 - (C_FSPL + 20*log10(2400.0) + 20*np.log10(d3/1000.0) + 10.0*blocked_mat(P, C))
    d2g = hav_m(np.radians(G01[0]), np.radians(G01[1]), np.radians(C[:, 0]), np.radians(C[:, 1]))
    d3g = np.sqrt(d2g**2 + (G01[2]-C[:, 2])**2)
    Gm = np.array([G01])
    prg = 36.0 - (C_FSPL + 20*log10(2400.0) + 20*np.log10(d3g/1000.0)
                  + 10.0*blocked_mat(Gm, C)[0])
    ok = (pra >= P_THR) & (prg[None, :] >= P_THR)
    return ok, pra, prg

_no = pd.read_csv(os.path.join(BASE, '预处理输出', 'Q1_节点_预处理.csv'))
_o = _no[_no['编号'] == 'O01'].iloc[0]
G01 = (float(_o['经度']), float(_o['纬度']), float(_o['海拔(m)']) + 20.0)

# ---- 候选网格（最近像元高程+300，12km内） ----
cand = []
for lo in np.arange(float(lons.min()), float(lons.max())+1e-9, GRID):
    for la in np.arange(float(lats.min()), float(lats.max())+1e-9, GRID):
        e = dem[int(np.argmin(np.abs(lats-la))), int(np.argmin(np.abs(lons-lo)))]
        if np.isfinite(e) and hav(G01[0], G01[1], lo, la) < R_MAX:
            cand.append((lo, la, e+HOVER_ABOVE))
cand = np.array(cand)
print('候选悬停点:', len(cand))

sched = pd.read_csv(os.path.join(BASE, 'Q3_两架方案_中继排程.csv'))
froz  = pd.read_csv(os.path.join(BASE, 'Q3_两架方案_逐点通信验证.csv'), encoding='utf-8-sig')
RC = {r['任务ID']: (float(r['经度']), float(r['纬度']), float(r['海拔m']),
                  float(r['到岗时刻s']), float(r['离岗时刻s'])) for _, r in sched.iterrows()}

# ---- 7个时间时段 ----
segments = []
for mid, g in froz.groupby('保障中继'):
    g = g.sort_values('t').reset_index(drop=True)
    edges = [0] + list(np.where(g['t'].diff() > GAP_CLUSTER)[0]) + [len(g)]
    for si, (a, b) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        segments.append((mid, si, g.iloc[a:b].reset_index(drop=True)))
print('时间时段数:', len(segments))

rec_rows = []
summary = []
missing = []
for mid, si, sg in segments:
    P = sg[['lon', 'lat', 'h']].values
    Cf = np.array([RC[mid][:3]])
    okf, pr1f, pr2f = cover(P, Cf)
    frozen_coord_ok = bool(okf[:, 0].all())
    if frozen_coord_ok:
        Cuse, src = Cf[0], '冻结排程表坐标'
        ok, pr1, pr2 = okf[:, 0], pr1f[:, 0], float(pr2f[0])
    else:
        # 缺失悬停点：局部走廊内独立贪心重建
        msk = ((cand[:, 0] >= sg['lon'].min()-0.08) & (cand[:, 0] <= sg['lon'].max()+0.08)
               & (cand[:, 1] >= sg['lat'].min()-0.08) & (cand[:, 1] <= sg['lat'].max()+0.08))
        Cl = cand[msk]
        cok, cpr1, cpr2 = cover(P, Cl)
        bi = int(np.argmax(cok.sum(axis=0)))
        Cuse, src = Cl[bi], '独立重建（冻结文件缺失）'
        ok, pr1, pr2 = cok[:, bi], cpr1[:, bi], float(cpr2[bi])
        missing.append(dict(任务ID=mid, 时段=si,
                            t起=round(float(sg['t'].min()), 1), t止=round(float(sg['t'].max()), 1),
                            重建经度=round(float(Cuse[0]), 5), 重建纬度=round(float(Cuse[1]), 5),
                            重建海拔m=round(float(Cuse[2]), 1),
                            排程表坐标覆盖率=round(float(okf[:, 0].mean()), 4),
                            覆盖点数=int(cok[:, bi].sum()), 时段点数=len(sg)))
    for j in range(len(sg)):
        rec_rows.append((float(sg.iat[j, 0]), sg.iat[j, 4], mid, si, src,
                         round(float(pr1[j]), 2), round(float(pr2), 2), int(bool(ok[j]))))
    summary.append(dict(任务ID=mid, 时段=si, 悬停点来源=src,
                        悬停经度=round(float(Cuse[0]), 5), 悬停纬度=round(float(Cuse[1]), 5),
                        悬停海拔m=round(float(Cuse[2]), 1),
                        t起=round(float(sg['t'].min()), 1), t止=round(float(sg['t'].max()), 1),
                        时段点数=len(sg), 核验通过=int(ok.sum()),
                        通过率=round(float(ok.mean()), 6),
                        TR中继最弱Pr=round(float(pr1.min()), 2),
                        中继G01最弱Pr=round(float(pr2), 2)))

res = pd.DataFrame(rec_rows, columns=['t', '架次', '保障中继', '时段', '悬停点来源',
                                      'TR中继弱向Pr', '中继G01弱向Pr', '核验通过'])
res.to_csv(os.path.join(OUT, 'Q3_重建_冻结方案逐点核验.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame(summary).to_csv(os.path.join(OUT, 'Q3_重建_冻结方案核验汇总.csv'),
                             index=False, encoding='utf-8-sig')
pd.DataFrame(missing).to_csv(os.path.join(OUT, 'Q3_重建_缺失悬停点.csv'),
                             index=False, encoding='utf-8-sig')
print(pd.DataFrame(summary).to_string(index=False))
if missing:
    print('\n缺失悬停点（独立重建）:')
    print(pd.DataFrame(missing).to_string(index=False))
print('-'*70)
print('总点数 %d，核验通过 %d，覆盖率 %.4f%%'
      % (len(res), int(res['核验通过'].sum()), 100.0*res['核验通过'].mean()))
# 窗口检查
win_bad = 0
for mid, si, sg in segments:
    w0, w1 = RC[mid][3], RC[mid][4]
    win_bad += int(((sg['t'] < w0) | (sg['t'] > w1)).sum())
print('超出排程表服务窗的点数:', win_bad, '（多悬停架次的窗为全包络窗）')
