# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。AI工具名称：DeepSeek；版本/型号：V4.1-Flash；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；版本发布日期：2026年9月10日。

"""
Q1 精确求解脚本（按核查意见全面修正）
1) 巡航海拔：航段与DEM栅格像元相交遍历（相邻中心中点为边界），取相交像元最高高程+50m
2) 服务区作业高度 = 地面海拔+30m；O01 作业高度 = 地面海拔
3) 能耗：E = E_use*d*[1/Lg(q)+1/Lg0] + 9.81/3600000/eta_c*[(m0+q)(H-h_O)+m0(H-h_S-30)]
   单位统一为 kWh；下降无附加能耗
4) 时间：t = (2H-h_O-h_S-30)*(1/v_up+1/v_dn) + 2d/v + t_prep + t_hand0 + n*(t_load+t_hand1)
5) 组批：枚举每服务区可行货箱子集×机型（架次级选型，允许同区混合机型），
   集合划分DP按字典序最小化 (架次数, 总能耗, 累计作业时间)
6) 最大安全载荷：先判空载往返是否满足返航余量，不满足标记"航线不可达"（m_safe=None）
7) 敏感性：逐返航余量(10%~40%)重新精确组批，报告架次数/能耗/时间/不可达服务区
输入：对比\数据 + 对比\预处理输出
输出：Q1_精确_最大安全载荷.csv、Q1_精确_组批方案.csv、Q1_精确_eta敏感性.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from math import radians, sin, cos, sqrt, atan2
from functools import lru_cache

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, '数据')
PREP = os.path.join(_HERE, '预处理输出')
OUT = _HERE
DEM_MAT = DATA + r'\镇龙乡地理空间数据\镇龙乡及周边地理数据\数字高程模型数据（DEM）\镇龙乡及周边30米DEM.mat'
G = 9.81

# ---------- 1. DEM ----------
m = loadmat(DEM_MAT)
dem = np.array(m['dem'], dtype=float)
lons = np.array(m['longitude']).ravel()
lats = np.array(m['latitude']).ravel()
dem[dem < 0] = np.nan
print(f'DEM: {dem.shape}, 高程范围 [{np.nanmin(dem):.1f}, {np.nanmax(dem):.1f}] m')

def at(lon, lat):
    i = int(np.argmin(np.abs(lats - lat)))
    j = int(np.argmin(np.abs(lons - lon)))
    return dem[i, j]

def cruise_exact(lon1, lat1, lon2, lat2):
    """遍历航段与所有DEM像元（相邻中心中点作边界）的交点，取最高地面高程+50。"""
    ts = [0.0, 1.0]
    for axis, a, b in [(lons, lon1, lon2), (lats, lat1, lat2)]:
        bounds = (axis[:-1] + axis[1:]) / 2.0
        t = (bounds - a) / (b - a)
        ts.extend(t[(t > 0) & (t < 1)].tolist())
    ts = sorted(set(ts))
    samples = [0.0, 1.0] + [(a + b) / 2.0 for a, b in zip(ts, ts[1:])]
    mx = max(at(lon1 + t * (lon2 - lon1), lat1 + t * (lat2 - lat1)) for t in samples)
    return mx + 50.0

def cruise_sampled(lon1, lat1, lon2, lat2, d):
    """原30m等距采样口径（仅作对比）。"""
    n = max(2, min(2000, int(d / 30)))
    mx = max(at(lon1 + t * (lon2 - lon1), lat1 + t * (lat2 - lat1))
             for t in np.linspace(0, 1, n + 1))
    return mx + 50.0

# ---------- 2. 数据 ----------
nodes = pd.read_csv(os.path.join(PREP, 'Q1_节点_预处理.csv'))
cargo = pd.read_csv(os.path.join(PREP, 'Q1_货箱明细_预处理.csv'))
uav = pd.read_csv(os.path.join(PREP, 'Q2_机型参数_预处理.csv'))

o = nodes[nodes['编号'] == 'O01'].iloc[0]
O_LON, O_LAT, O_ALT = float(o['经度']), float(o['纬度']), float(o['海拔(m)'])

def haversine(lon1, lat1, lon2, lat2):
    lon1_r, lat1_r, lon2_r, lat2_r = map(radians, [lon1, lat1, lon2, lat2])
    a = sin((lat2_r - lat1_r) / 2) ** 2 + cos(lat1_r) * cos(lat2_r) * sin((lon2_r - lon1_r) / 2) ** 2
    return 6371000 * 2 * atan2(sqrt(a), sqrt(1 - a))

sa_rows = nodes[nodes['编号'] != 'O01'].copy().reset_index(drop=True)
sa_rows['d'] = sa_rows.apply(lambda r: haversine(O_LON, O_LAT, float(r['经度']), float(r['纬度'])), axis=1)

pps = {}
for _, u in uav.iterrows():
    pps[u['机型编号']] = dict(
        m0=float(u['空载总质量(kg)']), Q=float(u['最大载货质量(kg)']),
        V=float(u['可用装载体积(m³)']), v=float(u['巡航速度(m/s)']),
        v_up=float(u['最大爬升速度(m/s)']), v_dn=float(u['最大下降速度(m/s)']),
        L0=float(u['空载标准航程(m)']), LF=float(u['满载标准航程(m)']),
        E=float(u['电池可用能量(kWh)']), rho=float(u['返航电量下限(%)']) / 100.0,
        eta=float(u['爬升能耗效率']),
        t_prep=float(u['工位准备时间(s)']), t_load=float(u['每箱装载时间(s)']),
        t_hand0=float(u['基础交接时间(s)']), t_hand1=float(u['每箱增加交接时间(s)']))

# ---------- 3. 题目规则能耗/时间（单位统一kWh） ----------
def Lg(pp, q):
    return pp['L0'] - (pp['L0'] - pp['LF']) * (q / pp['Q']) ** 1.5

def energy(pp, q, d, H, h_sa):
    """单架次往返总能耗 kWh：去程载货q + 返程空载 + 两次爬升附加（下降无附加）。"""
    out = pp['E'] * d / Lg(pp, q)
    ret = pp['E'] * d / pp['L0']
    up1 = 9.81 * (pp['m0'] + q) * (H - O_ALT) / (pp['eta'] * 3600000)
    up2 = 9.81 * pp['m0'] * (H - h_sa - 30) / (pp['eta'] * 3600000)
    return out + ret + up1 + up2

def trip_time(pp, q, d, H, h_sa, n_boxes):
    t_flight = (2 * H - O_ALT - h_sa - 30) * (1 / pp['v_up'] + 1 / pp['v_dn']) + 2 * d / pp['v']
    return t_flight + pp['t_prep'] + pp['t_hand0'] + n_boxes * (pp['t_load'] + pp['t_hand1'])

# ---------- 4. 45组最大安全载荷（exact地形；空载不可达→None） ----------
msafe_rows = []
for _, r in sa_rows.iterrows():
    H = cruise_exact(O_LON, O_LAT, float(r['经度']), float(r['纬度']))
    for mt in ['A', 'B', 'C']:
        pp = pps[mt]
        if energy(pp, 0.0, r['d'], H, float(r['海拔(m)'])) > (1 - pp['rho']) * pp['E']:
            m_safe = None          # 航线不可达
        else:
            lo, hi = 0.0, pp['Q']
            for _ in range(60):
                mid = (lo + hi) / 2
                if energy(pp, mid, r['d'], H, float(r['海拔(m)'])) <= (1 - pp['rho']) * pp['E']:
                    lo = mid
                else:
                    hi = mid
            m_safe = lo
        msafe_rows.append({'机型': mt, '服务区': r['编号'], 'm_safe(kg)': None if m_safe is None else round(m_safe, 4),
                           'm_max(kg)': pp['Q'], '巡航海拔_exact(m)': round(H, 3)})
df_msafe = pd.DataFrame(msafe_rows)
df_msafe.to_csv(os.path.join(OUT, 'Q1_精确_最大安全载荷.csv'), index=False, encoding='utf-8-sig')
print('\n=== 精确45组 m_safe（kg，None=空载往返不可达） ===')
print(df_msafe.pivot(index='机型', columns='服务区', values='m_safe(kg)').round(1).to_string())

# ---------- 5. 精确组批：子集×机型枚举 + 集合划分DP（允许同区混合机型） ----------
def exact_plan(sa, H, rho=0.2):
    """返回该服务区的字典序最优 (架次数, 能耗, 秒数, 架次列表)；不可行返回None。"""
    boxes = cargo[cargo['服务区编号'] == sa].copy().reset_index(drop=True)
    n = len(boxes)
    full = (1 << n) - 1
    if n == 0:
        return (0, 0.0, 0.0, [])
    W = [float(b['单箱质量(kg)']) for _, b in boxes.iterrows()]
    V = [float(b['单箱体积(m³)']) for _, b in boxes.iterrows()]
    ID = [b['货箱序号'] for _, b in boxes.iterrows()]
    wsum = [0.0] * (full + 1); vsum = [0.0] * (full + 1)
    for mask in range(1, full + 1):
        bit = mask & -mask
        i = bit.bit_length() - 1
        wsum[mask] = wsum[mask ^ bit] + W[i]
        vsum[mask] = vsum[mask ^ bit] + V[i]
    opts = {}          # mask -> (能耗, 秒数, 机型)
    for mask in range(1, full + 1):
        for g, pp in pps.items():
            if wsum[mask] <= pp['Q'] + 1e-9 and vsum[mask] <= pp['V'] + 1e-9:
                e = energy(pp, wsum[mask], sa_rows[sa_rows['编号'] == sa]['d'].iloc[0], H,
                           float(sa_rows[sa_rows['编号'] == sa]['海拔(m)'].iloc[0]))
                if e <= (1 - rho) * pp['E'] + 1e-9:
                    t = trip_time(pp, wsum[mask], sa_rows[sa_rows['编号'] == sa]['d'].iloc[0], H,
                                  float(sa_rows[sa_rows['编号'] == sa]['海拔(m)'].iloc[0]), mask.bit_count())
                    item = (e, t, g)
                    if mask not in opts or item < opts[mask]:
                        opts[mask] = item
    @lru_cache(None)
    def dp(mask):
        if mask == 0:
            return (0, 0.0, 0.0, ())
        if mask in opts:
            e, t, g = opts[mask]
            return (1, e, t, ((mask, g),))
        first = mask & -mask
        sub = mask
        best = (999, float('inf'), float('inf'), ())
        while sub:
            if (sub & first) and sub in opts:
                e, t, g = opts[sub]
                tail = dp(mask ^ sub)
                cand = (1 + tail[0], e + tail[1], t + tail[2], ((sub, g),) + tail[3])
                if cand[:3] < best[:3]:
                    best = cand
            sub = (sub - 1) & mask
        return best
    best = dp(full)
    if best[0] >= 999:
        return None
    d_sa = sa_rows[sa_rows['编号'] == sa]['d'].iloc[0]
    h_sa = float(sa_rows[sa_rows['编号'] == sa]['海拔(m)'].iloc[0])
    trips = []
    for mask, g in best[3]:
        pp = pps[g]
        box_ids = [ID[i] for i in range(n) if mask >> i & 1]
        q = wsum[mask]; e = energy(pp, q, d_sa, H, h_sa)
        trips.append({'服务区': sa, '机型': g, '货箱编号': ';'.join(map(str, box_ids)), '箱数': len(box_ids),
                      '载重kg': round(q, 4), '体积m3': round(vsum[mask], 4),
                      '能耗kWh': round(e, 4), '返回SOC': round(1 - e / pp['E'], 4),
                      '巡航海拔m': round(H, 3), '累计作业秒': round(trip_time(pp, q, d_sa, H, h_sa, len(box_ids)), 2)})
    return (best[0], best[1], best[2], trips)

print('\n=== 精确组批方案（exact地形，ρ=20%，混合机型开放） ===')
all_trips = []
total_n = total_E = total_T = 0
for _, r in sa_rows.iterrows():
    H = cruise_exact(O_LON, O_LAT, float(r['经度']), float(r['纬度']))
    pl = exact_plan(r['编号'], H)
    if pl is None:
        print(f'{r["编号"]}: 无可行解')
        continue
    for t in pl[3]:
        all_trips.append(t)
    total_n += pl[0]; total_E += pl[1]; total_T += pl[2]
    print(f'{r["编号"]}: {pl[0]}架次, {pl[1]:.4f} kWh, {pl[2]:.1f}s, 机型={[x["机型"] for x in pl[3]]}')
df_trips = pd.DataFrame(all_trips)
df_trips.to_csv(os.path.join(OUT, 'Q1_精确_组批方案.csv'), index=False, encoding='utf-8-sig')
print(f'\n总架次数={total_n}, 总能耗={total_E:.4f} kWh, 累计作业时间={total_T/3600:.4f} h')

# 逐箱唯一性与约束审计
ids = [i for t in all_trips for i in str(t['货箱编号']).split(';')]
print(f'审计: 分配货箱 {len(ids)} 个, 唯一 {len(set(ids))} 个, 全部80箱覆盖: {set(ids) == set(map(str, cargo["货箱序号"]))}')

# 逐架次明细审计：载重<=m_safe、体积<=Vmax、能耗<=可用能量（返回SOC>=rho）
msafe_map = {(r['机型'], r['服务区']): (None if r['m_safe(kg)'] is None else float(r['m_safe(kg)']))
             for _, r in df_msafe.iterrows()}
audit_rows = []
viol = []
for t in all_trips:
    ms = msafe_map[(t['机型'], t['服务区'])]
    pp = pps[t['机型']]
    ok_load = ms is not None and t['载重kg'] <= ms + 1e-9
    ok_vol = t['体积m3'] <= pp['V'] + 1e-9
    ok_ene = t['返回SOC'] >= pp['rho'] - 1e-9
    if not (ok_load and ok_vol and ok_ene):
        viol.append((t['服务区'], t['机型'], not ok_load, not ok_vol, not ok_ene))
    audit_rows.append({'服务区': t['服务区'], '机型': t['机型'], '箱数': t['箱数'],
                       '载重kg': t['载重kg'], 'm_safe(kg)': ms, '载重合规': ok_load,
                       '体积m3': t['体积m3'], '体积上限m3': pp['V'], '体积合规': ok_vol,
                       '能耗kWh': t['能耗kWh'], '可用能量kWh': round((1 - pp['rho']) * pp['E'], 4),
                       '返回SOC': t['返回SOC'], '余量下限': pp['rho'], '能耗合规': ok_ene})
df_audit = pd.DataFrame(audit_rows)
df_audit.to_csv(os.path.join(OUT, 'Q1_精确_逐架次审计.csv'), index=False, encoding='utf-8-sig')
print(f'逐架次审计: 共{len(audit_rows)}架次, 违规{len(viol)}项')
for v in viol:
    print(f'  违规: {v}')

# ---------- 6. 敏感性：逐余量重新精确组批 ----------
print('\n=== 返航余量敏感性（逐余量重新组批，exact地形） ===')
sens = []
for rho in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    total_n = total_E = total_T = 0
    bad = []
    for _, r in sa_rows.iterrows():
        H = cruise_exact(O_LON, O_LAT, float(r['经度']), float(r['纬度']))
        pl = exact_plan(r['编号'], H, rho)
        if pl is None:
            bad.append(r['编号'])
        else:
            total_n += pl[0]; total_E += pl[1]; total_T += pl[2]
    sens.append({'返航余量': rho, '可行': not bad, '不可达服务区': ';'.join(bad) if bad else '',
                 '最少架次数': total_n if not bad else None,
                 '总能耗kWh': round(total_E, 4) if not bad else None,
                 '累计作业时间h': round(total_T / 3600, 4) if not bad else None})
df_sens = pd.DataFrame(sens)
df_sens.to_csv(os.path.join(OUT, 'Q1_精确_eta敏感性.csv'), index=False, encoding='utf-8-sig')
print(df_sens.to_string(index=False))
