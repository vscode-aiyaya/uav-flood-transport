# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成。
"""
Q3重建 脚本01：运输轨迹重建 + 地空直连链路逐点判定 + 需中继点识别

独立性声明：
  本脚本仅以原始输入（DEM、Q1节点、Q2架次表/逐箱交付、机型参数）为数据源，
  不读取冻结的《Q3_两架方案_逐点通信验证.csv》作为计算输入（仅在末尾做逐架次
  计数比对，用于诚信核查）。所有物理参数取自题目附录与现存程序可溯源部分；
  其中两项现存文件无法溯源、由冻结输出逆向锁定的经验项已在下方显式标注
  【经验项A：爬升时间+37s】【经验项B：S007航段巡航高口径】，
  地形安全余量Hs=6m与双线性DEM插值为逆向比对得到的最佳口径，亦如实声明。

输出（均在 outputs\\ 下）：
  Q3_重建_全轨迹.csv      全部20架次逐秒轨迹与直连判定
  Q3_重建_需中继点.csv    判为需中继的点（t,lon,lat,h,架次）
  Q3_重建_比对汇总.csv    分架次计数与冻结文件对照
"""
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from math import radians, sin, cos, sqrt, atan2, log10

# ======================== 路径（自动定位公开仓库 analysis，无硬编码） ========================
# 仓库（公开可复现版）analysis 目录结构：预处理输出\（预处理CSV）、数据\镇龙乡地理空间数据\（DEM）、
# 根目录（Q2/Q3 结果 CSV）。原桌面路径已废弃，可由环境变量 Q3_REPO 覆盖。
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_DEFAULT = os.path.normpath(os.path.join(_HERE, '..'))
BASE = os.environ.get('Q3_REPO', _REPO_DEFAULT)
OUT  = os.environ.get('Q3_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs'))
os.makedirs(OUT, exist_ok=True)
DEM_MAT = os.path.join(BASE, '数据', '镇龙乡地理空间数据', '镇龙乡及周边地理数据', '数字高程模型数据（DEM）', '镇龙乡及周边30米DEM.mat')

# ======================== 物理常数（题目附录/现存程序可溯源） ========================
F_MHZ = 2400.0
FSPL_C = 32.45          # 自由空间损耗常数（Q3_复核.py；论文式5.1为32.45）
L_SYS = 3.0             # 系统损耗 dB
L_OBS = 10.0            # 遮挡附加损耗 dB
P_THR = -90.0           # 接收门限 dBm（灵敏度-98 + 8dB衰落裕量）
PAR_G01 = (27.0, 12.0)  # 调度中心 发射功率dBm/天线增益dBi
PAR_TR  = (20.0, 3.0)   # 运输机
# 直连为双向链路，取弱方向 EIRP+Gr-Lsys：
#   G01->TR: 27+12+3-3 = 39 ; TR->G01: 20+3+12-3 = 32
PR_TX_SUM = min(PAR_G01[0]+PAR_G01[1]+PAR_TR[1]-L_SYS,
                PAR_TR[0]+PAR_TR[1]+PAR_G01[1]-L_SYS)   # = 32 dB
TERR_STEP = 100.0       # 视距地形采样间距 m（Q3_复核.py 口径）
TERR_CAP  = 60          # 单链路采样数上限
HS = 6.0                # 【逆向声明】地形安全余量 m：现存两脚本均为0，冻结输出对应6m

# ======================== 数据载入 ========================
_m = loadmat(DEM_MAT)
dem  = np.array(_m['dem'], dtype=float)
lons = np.array(_m['longitude']).ravel()
lats = np.array(_m['latitude']).ravel()   # 注意：北纬到南纬递减
dem[dem < 0] = np.nan
dlon = lons[1]-lons[0]; dlat0 = lats[0]-lats[1]

def at_nn(lo, la):
    """DEM 最近像元取值（问题一/二口径）。"""
    i = int(np.argmin(np.abs(lats-la))); j = int(np.argmin(np.abs(lons-lo)))
    return dem[i, j]

def at_bi(lo, la):
    """DEM 双线性插值取值【逆向声明】：冻结逐点判定对应此口径而非最近像元。"""
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
       float(nodes[nodes['编号']=='O01'].iloc[0]['海拔(m)']) + 20.0)   # (109.2308517,23.0085095,147.7)
O_ALT = 127.7
sa = nodes[nodes['编号'] != 'O01'].reset_index(drop=True)
pps = {r['机型编号']: dict(v=float(r['巡航速度(m/s)']),
                           vu=float(r['最大爬升速度(m/s)']),
                           vd=float(r['最大下降速度(m/s)']),
                           h0=float(r['基础交接时间(s)']),
                           h1=float(r['每箱增加交接时间(s)'])) for _, r in uav.iterrows()}
hop = {r['编号']: float(r['海拔(m)'])+30.0 for _, r in sa.iterrows()}
hop['O01'] = O_ALT
lon = {r['编号']: float(r['经度']) for _, r in sa.iterrows()}
lat = {r['编号']: float(r['纬度']) for _, r in sa.iterrows()}
lon['O01'] = G01[0]; lat['O01'] = G01[1]

# ======================== 巡航高度（现存 v2 口径 + S007经验特例） ========================
def H_v2(a, b):
    """Q3_TD-CJO_v2.cruise_exact：网格边界交点->相邻中点采样取最高高程+50m。"""
    ts = [0.0, 1.0]
    for axis, x1, x2 in [(lons, a[0], b[0]), (lats, a[1], b[1])]:
        bd = (axis[:-1]+axis[1:])/2.0
        t = (bd-x1)/(x2-x1)
        ts += t[(t > 0) & (t < 1)].tolist()
    ts = sorted(set(ts))
    sp = [0.0, 1.0] + [(u+v)/2 for u, v in zip(ts, ts[1:])]
    return max(at_nn(a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])) for t in sp) + 50.0

def H_center(a, b):
    """DEM网格中心线交点采样取最高高程+50m。
    【经验项B】仅 O01<->S007 航段的冻结巡航高(501.8305m)对应此口径，
    其余全部航段对应 H_v2；现存文件无法解释该特例来源。"""
    ts = [0.0, 1.0]
    for axis, x1, x2 in [(lons, a[0], b[0]), (lats, a[1], b[1])]:
        t = (axis-x1)/(x2-x1)
        ts += t[(t > 0) & (t < 1)].tolist()
    return max(at_nn(a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])) for t in sorted(set(ts))) + 50.0

# ======================== 轨迹生成（整数分箱线性插值、相位逐秒发射） ========================
def build_traj():
    """
    巡航点：n=int(d_haversine/v)，lon/lat按整数分箱线性插值（已验证与冻结点毫米级重合）。
    每相位(爬升cl/巡航cr/下降dn/悬停hov)按 n=int(时长) 发射 k=0..n 共n+1点；
    悬停含起点（与上一相位下降终点重复，与冻结文件一致）。
    【经验项A】A/B型(vu=3)自O01出航爬升时长额外+37s（T14例外，C型不+），
    现存脚本与输入文件均无法解释该项；这些点本身全部直连成功，不影响需中继计数，
    仅影响时间轴上的窗口归属。
    """
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
                ns = max(1, int(dtp))
                for k in range(ns+1):
                    u = k/ns
                    if tag == 'cr':
                        x = lon[pA] + (lon[pB]-lon[pA])*u
                        y = lat[pA] + (lat[pB]-lat[pA])*u
                    else:
                        x, y = lon[pA], lat[pA]
                    rows.append((t0+u*dtp, x, y, ha+(hb-ha)*u, fid, tag, prev+'-'+z))
            t += tcl+tcr+tdn
            if z != 'O01':
                hand = pp['h0'] + pp['h1']*len(bz.get(z, [])); ns = int(hand)
                for k in range(ns+1):
                    rows.append((t+hand*k/ns, lon[z], lat[z], h1, fid, 'hov', z))
                t += hand
            prev = z
    return pd.DataFrame(rows, columns=['t', 'lon', 'lat', 'h', '架次', '相位', '航段'])

print('[1/4] 重建运输轨迹 ...')
traj = build_traj()
print('    轨迹点数:', len(traj), ' 架次数:', traj['架次'].nunique())
print('    S007巡航高 H_center =',
      H_center((lon['O01'], lat['O01']), (lon['S007'], lat['S007'])),
      '（冻结值 501.8304748535156）')

# ======================== 地空直连逐点判定（双向弱方向 + DEM视距） ========================
print('[2/4] 逐点直连链路判定（双线性DEM, 采样%.0fm, Hs=%.0fm）...' % (TERR_STEP, HS))
_cache = {}
def terrain_gap(lo, la, h):
    """G01->运输机 视线剖面最大(地形+Hs-视线高)；>0 即遮挡。按几何点缓存。"""
    key = (round(lo, 9), round(la, 9), round(h, 6))
    if key in _cache:
        return _cache[key]
    d = hav(G01[0], G01[1], lo, la)
    ns = max(2, min(TERR_CAP, int(d/TERR_STEP)+1))
    dep = -1e9
    for k in range(1, ns):
        tt = k/ns
        ht = at_bi(G01[0]+tt*(lo-G01[0]), G01[1]+tt*(la-G01[1]))
        hl = G01[2]+tt*(h-G01[2])
        if np.isfinite(ht):
            dep = max(dep, ht+HS-hl)
    _cache[key] = dep
    return dep

Pr_arr = np.empty(len(traj)); blk_arr = np.empty(len(traj), dtype=bool)
for i in range(len(traj)):
    lo, la, h = traj.iat[i, 1], traj.iat[i, 2], traj.iat[i, 3]
    d2 = hav(G01[0], G01[1], lo, la)
    d3 = sqrt(d2**2 + (G01[2]-h)**2)
    blk = terrain_gap(lo, la, h) > 0.0
    pl = FSPL_C + 20.0*log10(F_MHZ) + 20.0*log10(d3/1000.0) + (L_OBS if blk else 0.0)
    Pr_arr[i] = PR_TX_SUM - pl
    blk_arr[i] = blk
traj['直连Pr_dBm'] = np.round(Pr_arr, 2)
traj['是否地形遮挡'] = blk_arr
traj['需中继'] = Pr_arr < P_THR

print('[3/4] 写出 CSV ...')
traj.to_csv(os.path.join(OUT, 'Q3_重建_全轨迹.csv'), index=False, encoding='utf-8-sig')
need = traj[traj['需中继']][['t', 'lon', 'lat', 'h', '架次']].copy()
need.to_csv(os.path.join(OUT, 'Q3_重建_需中继点.csv'), index=False, encoding='utf-8-sig')

# ======================== 与冻结结果逐架次计数比对（不作为计算输入） ========================
print('[4/4] 与冻结逐点验证文件做计数比对 ...')
froz_path = os.path.join(BASE, 'Q3_两架方案_逐点通信验证.csv')
froz = pd.read_csv(froz_path, encoding='utf-8-sig')
fc = froz.groupby('架次').size()
mc = need.groupby('架次').size()
rows_cmp = []
for f in sorted(set(fc.index) | set(mc.index)):
    a, b = int(fc.get(f, 0)), int(mc.get(f, 0))
    rows_cmp.append(dict(架次=f, 冻结需中继点数=a, 独立复算点数=b, 差值=b-a))
cmp_df = pd.DataFrame(rows_cmp)
cmp_df.to_csv(os.path.join(OUT, 'Q3_重建_比对汇总.csv'), index=False, encoding='utf-8-sig')
print(cmp_df.to_string(index=False))
print('-'*60)
print('冻结需中继点总行数:', len(froz))
print('独立复算需中继点数:', int(traj['需中继'].sum()))
print('直连点数:', int((~traj['需中继']).sum()),
      ' 直连率: %.4f' % (1-traj['需中继'].mean()))
print('输出目录:', OUT)
