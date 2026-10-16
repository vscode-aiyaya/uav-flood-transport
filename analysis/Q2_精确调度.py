# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。AI工具名称：DeepSeek；版本/型号：V4.1-Flash；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；版本发布日期：2026年9月10日。

"""
Q2 受限任务结构枚举与贪心排程（修正版 v4）
物理层与Q1同口径：逐航段DEM巡航海拔、非线性等效航程Lg(q)、爬升附加、下降无附加
约束层：31箱硬时限逐箱核验；两阶段充电A/B/C=1800/2400/3000s；8架编号无人机；逐块电池
搜索层（v4，明确搜索范围，非全问题精确解）：
  - 固定部分：S001~S006/S010/S012/S013 沿用Q1精确架次组批（单区往返）；
    S007/S008/S014(/S015) 首批2箱拆出独立任务；剩余箱与 S009/S011(/S015) 整区组成合并池
  - 合并池：按"第一个未分配单元+其余任意可行子集"枚举全部集合划分，
    每组 ≤3 单元、≤80kg、≤0.25m³；多点路线仅用C型、逐段剩余载荷计算
  - 机型：A/B/C 按容量可行性全部进入选项，由排程（紧迫度优先贪心）按资源可用性选择
  - 排程：任务按硬截止时间升序，每任务在可行机型选项中选取最早可起飞的无人机×电池组合
结果层：四指标（架次数/总能耗/Cmax/软延误）记录全部可行方案，按架次×能耗筛选非支配解
"""
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from math import radians, sin, cos, sqrt, atan2
from itertools import permutations

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, '数据')
PREP = os.path.join(_HERE, '预处理输出')
OUT = _HERE
DEM_MAT = DATA + r'\镇龙乡地理空间数据\镇龙乡及周边地理数据\数字高程模型数据（DEM）\镇龙乡及周边30米DEM.mat'
G = 9.81

m = loadmat(DEM_MAT)
dem = np.array(m['dem'], dtype=float)
lons = np.array(m['longitude']).ravel()
lats = np.array(m['latitude']).ravel()
dem[dem < 0] = np.nan

def at(lon, lat):
    i = int(np.argmin(np.abs(lats - lat)))
    j = int(np.argmin(np.abs(lons - lon)))
    return dem[i, j]

def cruise_exact(lon1, lat1, lon2, lat2):
    ts = [0.0, 1.0]
    for axis, a, b in [(lons, lon1, lon2), (lats, lat1, lat2)]:
        bounds = (axis[:-1] + axis[1:]) / 2.0
        t = (bounds - a) / (b - a)
        ts.extend(t[(t > 0) & (t < 1)].tolist())
    ts = sorted(set(ts))
    samples = [0.0, 1.0] + [(a + b) / 2.0 for a, b in zip(ts, ts[1:])]
    mx = max(at(lon1 + t * (lon2 - lon1), lat1 + t * (lat2 - lat1)) for t in samples)
    return mx + 50.0

def hav(lon1, lat1, lon2, lat2):
    lon1_r, lat1_r, lon2_r, lat2_r = map(radians, [lon1, lat1, lon2, lat2])
    a = sin((lat2_r - lat1_r) / 2) ** 2 + cos(lat1_r) * cos(lat2_r) * sin((lon2_r - lon1_r) / 2) ** 2
    return 6371000 * 2 * atan2(sqrt(a), sqrt(1 - a))

nodes = pd.read_csv(os.path.join(PREP, 'Q1_节点_预处理.csv'))
cargo = pd.read_csv(os.path.join(PREP, 'Q1_货箱明细_预处理.csv'))
uav = pd.read_csv(os.path.join(PREP, 'Q2_机型参数_预处理.csv'))
batt = pd.read_csv(os.path.join(PREP, 'Q2_电池库存_预处理.csv'))

o = nodes[nodes['编号'] == 'O01'].iloc[0]
O_LON, O_LAT, O_ALT = float(o['经度']), float(o['纬度']), float(o['海拔(m)'])
sa_rows = nodes[nodes['编号'] != 'O01'].copy().reset_index(drop=True)

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
battery = {r['机型编号']: (int(r['共享电池总数(组)']), float(r['等效完全充电时间(s)'])) for _, r in batt.iterrows()}
FLEET = {'A': 4, 'B': 2, 'C': 2}
BATS = {mt: battery[mt][0] for mt in FLEET}
TCHG = {mt: battery[mt][1] for mt in FLEET}

def Lg(pp, q):
    return pp['L0'] - (pp['L0'] - pp['LF']) * (q / pp['Q']) ** 1.5

# 作业高度：O01取地面海拔，服务区取地面海拔+30m（题目悬停/作业高度）
h_op = {r['编号']: float(r['海拔(m)']) + 30.0 for _, r in sa_rows.iterrows()}
h_op['O01'] = O_ALT
lon = {r['编号']: float(r['经度']) for _, r in sa_rows.iterrows()}
lat = {r['编号']: float(r['纬度']) for _, r in sa_rows.iterrows()}
lon['O01'], lat['O01'] = O_LON, O_LAT

pool_all = ['S007', 'S008', 'S009', 'S010', 'S011', 'S012', 'S013', 'S014', 'S015']
Hc = {}
for s in sa_rows['编号'].tolist():
    Hc[('O01', s)] = Hc[(s, 'O01')] = cruise_exact(O_LON, O_LAT, lon[s], lat[s])
for i in range(len(pool_all)):
    for j in range(i + 1, len(pool_all)):
        a, b = pool_all[i], pool_all[j]
        Hc[(a, b)] = Hc[(b, a)] = cruise_exact(lon[a], lat[a], lon[b], lat[b])
print('航段巡航海拔预计算完成')

def seg_time(pp, d, Hc_val, h0, h1):
    return max(Hc_val - h0, 0.0) / pp['v_up'] + d / pp['v'] + max(Hc_val - h1, 0.0) / pp['v_dn']

def seg_energy(pp, q, d, Hc_val, h0, h1):
    return pp['E'] * d / Lg(pp, q) + 9.81 * (pp['m0'] + q) * max(Hc_val - h0, 0.0) / (pp['eta'] * 3600000)

def task_option(seq, boxsets_by_zone, mt):
    pp = pps[mt]
    total_w = sum(cargo[cargo['货箱序号'] == b]['单箱质量(kg)'].iloc[0]
                  for z in seq[1:-1] for b in boxsets_by_zone[z])
    if total_w > pp['Q'] + 1e-9:
        return None
    vol = sum(cargo[cargo['货箱序号'] == b]['单箱体积(m³)'].iloc[0]
              for z in seq[1:-1] for b in boxsets_by_zone[z])
    if vol > pp['V'] + 1e-9:
        return None
    rem = total_w
    E = 0.0
    for k in range(len(seq) - 1):
        a, b = seq[k], seq[k + 1]
        d = hav(lon[a], lat[a], lon[b], lat[b])
        q = rem if b != 'O01' else 0.0
        E += seg_energy(pp, q, d, Hc[(a, b)], h_op[a], h_op[b])
        if b != 'O01':
            rem -= sum(cargo[cargo['货箱序号'] == bx]['单箱质量(kg)'].iloc[0] for bx in boxsets_by_zone[b])
    if E > (1 - pp['rho']) * pp['E'] + 1e-9:
        return None
    T = sum(seg_time(pp, hav(lon[seq[k]], lat[seq[k]], lon[seq[k+1]], lat[seq[k+1]]),
                     Hc[(seq[k], seq[k+1])], h_op[seq[k]], h_op[seq[k+1]]) for k in range(len(seq) - 1))
    n_total = sum(len(v) for v in boxsets_by_zone.values())
    T += pp['t_prep'] + pp['t_load'] * n_total
    for z in seq[1:-1]:
        T += pp['t_hand0'] + pp['t_hand1'] * len(boxsets_by_zone[z])
    rel_arr = {}
    t_acc = pp['t_prep'] + pp['t_load'] * n_total
    for k in range(1, len(seq) - 1):
        t_acc += seg_time(pp, hav(lon[seq[k-1]], lat[seq[k-1]], lon[seq[k]], lat[seq[k]]),
                          Hc[(seq[k-1], seq[k])], h_op[seq[k-1]], h_op[seq[k]])
        rel_arr[seq[k]] = t_acc + pp['t_hand0'] + pp['t_hand1'] * len(boxsets_by_zone[seq[k]])
        t_acc += pp['t_hand0'] + pp['t_hand1'] * len(boxsets_by_zone[seq[k]])
    boxes = [b for z in seq[1:-1] for b in boxsets_by_zone[z]]
    return dict(mt=mt, E=E, T=T, rel_arr=rel_arr, boxes=boxes, n=len(boxes), total_w=total_w)

# ---------- 货箱时限 ----------
cargo['是否首批'] = cargo['是否首批'].astype(float) >= 0.5
hard_deadline = {}
for _, b in cargo.iterrows():
    dl = []
    if b['是否首批'] and not pd.isna(b['首批截止时间(s)']):
        dl.append(float(b['首批截止时间(s)']))
    if b['物资类型'] == '医疗物资' and not pd.isna(b['期望送达时间(s)']):
        dl.append(float(b['期望送达时间(s)']))
    if dl:
        hard_deadline[b['货箱序号']] = min(dl)
soft_exp = {b['货箱序号']: float(b['期望送达时间(s)']) for _, b in cargo.iterrows()}
print(f'硬时限箱数: {len(hard_deadline)} (应为31)')

box_by_zone = {z: [str(b) for b in cargo[cargo['服务区编号'] == z]['货箱序号']] for z in sa_rows['编号']}

def mk_task(zones, boxsets, mt_list):
    options = []
    for mt in mt_list:
        zb = {z: boxsets[z] for z in zones}
        op = task_option(['O01'] + zones + ['O01'], zb, mt)
        if op is not None:
            op['mt'] = mt
            options.append(op)
    if not options:
        return None
    return dict(zones=zones, seq=['O01'] + zones + ['O01'], options=options,
                deadline=min((hard_deadline[b] for z in zones for b in boxsets[z] if b in hard_deadline),
                             default=float('inf')))

# ---------- 大区固定任务（S015拆/不拆 双结构） ----------
q1 = pd.read_csv(os.path.join(OUT, 'Q1_精确_组批方案.csv'))
def build_big(split_s015):
    big = []
    for _, r in q1.iterrows():
        if r['服务区'] in ('S007', 'S008', 'S009', 'S011', 'S014', 'S015'):
            continue
        zb = {r['服务区']: [str(x).strip() for x in str(r['货箱编号']).split(';')]}
        t = mk_task([r['服务区']], zb, [r['机型']])
        assert t is not None, f'大区任务失败: {r["服务区"]} {r["机型"]}'
        big.append(t)
    fs_first = {}
    for z in ['S007', 'S008', 'S014'] + (['S015'] if split_s015 else []):
        fs = [b for b in box_by_zone[z] if b in hard_deadline]
        assert len(fs) == 2, (z, fs)
        fs_first[z] = fs
        t = mk_task([z], {z: fs}, ['A', 'B', 'C'])  # 机型不预设，由容量可行性与排程决定
        assert t is not None, f'首批快送失败: {z}'
        big.append(t)
    return big, fs_first

big_no, _ = build_big(False)
big_yes, fs_first_yes = build_big(True)
print(f'大区任务数: 不拆S015={len(big_no)}  拆S015={len(big_yes)}')

# ---------- 小区合并池（按S015拆/不拆） ----------
def build_units(split_s015):
    us = []
    for z in ['S009', 'S011'] + (['S015'] if not split_s015 else []):
        us.append((f'U_{z}', [z], {z: box_by_zone[z]}))
    fs_first = fs_first_yes if split_s015 else {'S007': None, 'S008': None, 'S014': None}
    for z in ['S007', 'S008', 'S014'] + (['S015'] if split_s015 else []):
        fs = [b for b in box_by_zone[z] if b in hard_deadline]
        rest = [b for b in box_by_zone[z] if b not in fs]
        us.append((f'r{z}', [z], {z: rest}))
    return us

units_no = build_units(False)
units_yes = build_units(True)
print(f'合并池单元数: 不拆={len(units_no)}  拆={len(units_yes)}')

def unit_w(unit):
    return sum(cargo[cargo['货箱序号'] == b]['单箱质量(kg)'].iloc[0] for bs in unit[2].values() for b in bs)

def unit_v(unit):
    return sum(cargo[cargo['货箱序号'] == b]['单箱体积(m³)'].iloc[0] for bs in unit[2].values() for b in bs)

def group_ok(us):
    w = sum(unit_w(u) for u in us)
    v = sum(unit_v(u) for u in us)
    return w <= 80 + 1e-9 and v <= 0.25 + 1e-9 and len(us) <= 3

def partitions(nds, cur, acc):
    """完整集合划分：第一个未分配单元 + 其余任意可行子集（非相邻单元可同组）"""
    if not nds:
        acc.append(list(cur))
        return
    first = nds[0]
    rest = nds[1:]
    # 方案1：first 单独成组
    partitions(rest, cur + [[first]], acc)
    # 方案2：first 与 rest 的任一非空子集同组（组须满足载重/体积/单元数限制）
    for k in range(1, 1 << len(rest)):
        g = [first] + [rest[i] for i in range(len(rest)) if k >> i & 1]
        if not group_ok(g):
            continue
        others = [rest[i] for i in range(len(rest)) if not (k >> i & 1)]
        partitions(others, cur + [g], acc)

def task_for_units(us):
    if len(us) == 1:
        u = us[0]
        w = unit_w(u)
        v = unit_v(u)
        cand = []
        for mt in ['A', 'B', 'C']:
            if w <= pps[mt]['Q'] + 1e-9 and v <= pps[mt]['V'] + 1e-9:
                cand.append(mt)
        return mk_task(u[1], u[2], cand)
    zones, boxsets = [], {}
    for u in us:
        zones += [z for z in u[1] if z not in zones]
        for z, bs in u[2].items():
            boxsets[z] = bs
    best = None
    for perm in permutations(zones):
        zb = {z: boxsets[z] for z in perm}
        op = task_option(['O01'] + list(perm) + ['O01'], zb, 'C')
        if op is None:
            continue
        if best is None or op['E'] < best[0]['E']:
            best = (op, list(perm))
    if best is None:
        return None
    op, perm_saved = best
    # 关键：保存被选中的访问顺序，输出/能耗/时刻均按同一顺序生成
    return dict(zones=zones, seq=['O01'] + perm_saved + ['O01'], options=[op],
                deadline=min((hard_deadline[b] for z in zones for b in boxsets[z] if b in hard_deadline),
                             default=float('inf')))

# ---------- 排程 ----------
def charge_time(s, Tfull):
    if s < 0.9:
        return Tfull * (0.65 * (0.9 - s) / 0.9 + 0.35)
    return Tfull * 0.35 * (1.0 - s) / 0.1

def schedule(tasks):
    uav_free = {f'{mt}{i}': 0.0 for mt in FLEET for i in range(1, FLEET[mt] + 1)}
    bat_free = {mt: [0.0] * BATS[mt] for mt in FLEET}
    remain = list(tasks)
    sched = []
    while remain:
        remain.sort(key=lambda t: t['deadline'])
        pick = remain.pop(0)
        best = None
        for opt in pick['options']:
            mt = opt['mt']
            uav_t = min(uav_free[f'{mt}{i}'] for i in range(1, FLEET[mt] + 1))
            bat_t = min(bat_free[mt])
            t_dep = max(uav_t, bat_t)
            if best is None or t_dep < best[0] - 1e-9:
                uav_id = min((f'{mt}{i}' for i in range(1, FLEET[mt] + 1)), key=lambda k: uav_free[k])
                bat_idx = bat_free[mt].index(min(bat_free[mt]))
                best = (t_dep, opt, mt, uav_id, bat_idx)
        t_dep, opt, mt, uav_id, bat_idx = best
        t_ret = t_dep + opt['T']
        s_end = 1.0 - opt['E'] / pps[mt]['E']  # 真实SOC（task_option已保证 ≥ rho）
        t_bat_ready = t_ret + charge_time(s_end, TCHG[mt])
        uav_free[uav_id] = t_ret
        bat_free[mt][bat_idx] = t_bat_ready
        arr = {z: t_dep + opt['rel_arr'][z] for z in opt['rel_arr']}
        hard_late = []
        for b in opt['boxes']:
            z = next(z for z in pick['zones'] if b in cargo[cargo['服务区编号'] == z]['货箱序号'].tolist())
            if b in hard_deadline and arr[z] > hard_deadline[b] + 1e-9:
                hard_late.append((b, hard_deadline[b], round(arr[z], 1)))
        sched.append(dict(zones=pick['zones'], seq=pick['seq'], mt=mt, uav=uav_id, bat=(mt, bat_idx + 1),
                          dep=t_dep, ret=t_ret, E=opt['E'], arr=arr, boxes=opt['boxes'],
                          hard_late=hard_late, bat_ready=t_bat_ready))
    Cmax = max(s['ret'] for s in sched)
    all_hard_late = [x for s in sched for x in s['hard_late']]
    soft_min = 0.0
    for s in sched:
        for b in s['boxes']:
            if b not in hard_deadline:
                z = next(z for z in s['zones'] if b in cargo[cargo['服务区编号'] == z]['货箱序号'].tolist())
                if s['arr'][z] > soft_exp[b] + 1e-9:
                    soft_min += (s['arr'][z] - soft_exp[b]) / 60.0
    return sched, Cmax, all_hard_late, soft_min

def plan_value(tasks):
    sched, Cmax, hard_late, soft_min = schedule(tasks)
    E = sum(s['E'] for s in sched)
    used_uav = len({s['uav'] for s in sched})
    return dict(N=len(tasks), E=E, Cmax=Cmax, hard_late=len(hard_late), soft_min=soft_min,
                used_uav=used_uav, sched=sched, hard_detail=hard_late)

# ---------- 枚举分组（双结构：S015拆/不拆） → 帕累托 ----------
results = []
for split_s015, big_tasks, units in [(False, big_no, units_no), (True, big_yes, units_yes)]:
    acc = []
    partitions(units, [], acc)
    for part in acc:
        tasks = list(big_tasks)
        ok = True
        for g in part:
            t = task_for_units(g)
            if t is None:
                ok = False
                break
            tasks.append(t)
        if not ok:
            continue
        r = plan_value(tasks)
        r['part'] = part
        r['split_s015'] = split_s015
        results.append(r)

print(f'全部方案: {len(results)}, 硬时限可行: {sum(1 for r in results if r["hard_late"] == 0)}')
feasible = [r for r in results if r['hard_late'] == 0]
pareto = []
for r in sorted(feasible, key=lambda x: (x['N'], x['E'])):
    if not any(p['N'] <= r['N'] and p['E'] <= r['E'] - 1e-9 and (p['N'] < r['N'] or p['E'] < r['E'] - 1e-9) for p in pareto):
        pareto.append(r)

print('=== 帕累托前沿（架次×能耗） ===')
for p in pareto:
    print(f'N={p["N"]}  E={p["E"]:.4f}kWh  Cmax={p["Cmax"]/3600:.3f}h  硬超时={p["hard_late"]}  软延误={p["soft_min"]:.1f}min  无人机={p["used_uav"]}  拆S015={p["split_s015"]}  part={p["part"]}')

df_all = pd.DataFrame([dict(N=r['N'], E=round(r['E'], 4), Cmax_h=round(r['Cmax'] / 3600, 4),
                             hard_late=r['hard_late'], soft_min=round(r['soft_min'], 2), used_uav=r['used_uav'],
                             split_s015=r['split_s015'], part=str(r['part'])) for r in results])
df_all.sort_values(['N', 'E']).to_csv(os.path.join(OUT, 'Q2_全部方案.csv'), index=False, encoding='utf-8-sig')
df_pf = pd.DataFrame([dict(N=p['N'], E=round(p['E'], 4), Cmax_h=round(p['Cmax'] / 3600, 4),
                            hard_late=p['hard_late'], soft_min=round(p['soft_min'], 2), used_uav=p['used_uav'],
                            split_s015=p['split_s015']) for p in pareto])
df_pf.to_csv(os.path.join(OUT, 'Q2_帕累托前沿.csv'), index=False, encoding='utf-8-sig')
print('保存 Q2_全部方案.csv / Q2_帕累托前沿.csv')
