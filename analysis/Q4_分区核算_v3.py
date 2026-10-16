# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。AI工具名称：DeepSeek；版本/型号：V4.1-Flash；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；版本发布日期：2026年9月10日。

"""
Q4 分区核算 v3（0836全篇复核修订基准）
[1] 中继任务时间线直接读取 Q3_两架方案_中继排程.csv（准备开始/周转完成/充电完成/悬停绝对海拔），
    不再重复相加海拔与离地、不再 max(0,..) 截断负准备（Q3排程本身已保证可达性）
[2] 中继服务对象由 Q3_两架方案_逐点通信验证.csv 逐点归属推导（避免旧手工映射）
[3] 完整占用区间：中继机[准备开始,周转完成]、中继组件[准备开始,充电完成]、
    运输机[准备开始,返航]、运输电池[准备开始,电池充满]
[4] 候选/均衡/字典序/零复制对照 与 v2 相同
"""
import os, json
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict

import os
BASE = os.path.dirname(os.path.abspath(__file__))
PREP = BASE + r'\预处理输出'

INV = {'uavA': 4, 'uavB': 2, 'uavC': 2, 'batA': 6, 'batB': 4, 'batC': 4,
       'relay': 2, 'relay_comp': 6}

def peak_concurrent(intervals):
    ev = []
    for s, e in intervals:
        ev.append((float(s), 1)); ev.append((float(e), -1))
    ev.sort(key=lambda x: (x[0], x[1]))
    cur = mx = 0
    for _, d in ev:
        cur += d; mx = max(mx, cur)
    return mx

def hav(lo1, la1, lo2, la2):
    lo1_r, la1_r, lo2_r, la2_r = map(radians, [lo1, la1, lo2, la2])
    a = sin((la2_r - la1_r) / 2) ** 2 + cos(la1_r) * cos(la2_r) * sin((lo2_r - lo1_r) / 2) ** 2
    return 6371000 * 2 * atan2(sqrt(a), sqrt(1 - a))

# ---------- 服务区 ----------
nodes = pd.read_csv(PREP + r'\Q1_节点_预处理.csv')
o = nodes[nodes['编号'] == 'O01'].iloc[0]
O_LON, O_LAT, O_ALT = float(o['经度']), float(o['纬度']), float(o['海拔(m)'])
sa_rows = nodes[nodes['编号'] != 'O01'].copy().reset_index(drop=True)
sa_info = {r['编号']: dict(lon=float(r['经度']), lat=float(r['纬度']), elev=float(r['海拔(m)'])) for _, r in sa_rows.iterrows()}
ALL_SA = sorted(sa_info.keys())
_q4old = json.load(open(BASE + r'\Q4_分区核算结果.json', encoding='utf-8'))
sa_demand = defaultdict(lambda: dict(boxes=0, mass=0, vol=0, first=0))
for row in _q4old.get('sa_table', []):
    sa_demand[row['sa']] = dict(boxes=int(row['boxes']), mass=float(row['mass']),
                                vol=float(row.get('vol', 0.0)), first=int(row.get('first', 0)))

# ---------- 运输任务 ----------
trips = pd.read_csv(BASE + r'\Q2_主方案20架次_架次表.csv')
sched = pd.read_csv(BASE + r'\Q2_主方案20架次_无人机时刻表.csv')
trip_meta = {r['架次ID']: dict(battery=r['电池'], bat_full=float(r['电池充满s']),
                               prep=float(r['准备开始s']), takeoff=float(r['起飞s']), ret=float(r['返航s']))
             for _, r in sched.iterrows()}
def sa_of_path(path):
    return [p for p in str(path).split('\u2192') if p not in ('O01', '') and p.strip()]
task_list = []
for _, r in trips.iterrows():
    tid = r['架次ID']; mt = trip_meta[tid]
    task_list.append(dict(tid=tid, uav=r['无人机'], mtype=r['机型'],
        sas=sa_of_path(r['路径']), prep=mt['prep'], takeoff=mt['takeoff'], ret=mt['ret'],
        battery=mt['battery'], bat_full=mt['bat_full'], energy=float(r['能耗kWh'])))

# ---------- 中继任务：完整时间线直接读冻结CSV；服务对象由逐点归属推导 ----------
relays = pd.read_csv(BASE + r'\Q3_两架方案_中继排程.csv')
perpt = pd.read_csv(BASE + r'\Q3_两架方案_逐点通信验证.csv')
relay_sas_of = defaultdict(set)   # 任务 -> 服务区集
for _, p in perpt.iterrows():
    w = p['保障中继']
    if not isinstance(w, str) or not w: continue
    tid_task = w.split('-', 1)[1] if '-' in w else w   # 'R01-簇1-1' -> '簇1-1'
    for t in task_list:
        if t['tid'] == p['架次']:
            relay_sas_of[tid_task].update(t['sas'])
relay_list = []
for _, r in relays.iterrows():
    rid = str(r['任务ID'])
    prep = float(r['准备开始s']); back = float(r['周转完成s']); chg = float(r['充电完成s'])
    relay_list.append(dict(rid=rid, machine=str(r['中继编号']),
        sas=sorted(relay_sas_of.get(rid, [])),
        arrive=float(r['到岗时刻s']), depart=float(r['离岗时刻s']),
        energy=float(r['架次能耗kWh']),
        h_abs=float(r['海拔m']), agl=float(r['离地高度m']),
        prep=prep, back=back, chg_done=chg,
        fl_go=float(r['巡航海拔m']) * 0.0 + 0.0, fl_back=0.0))
    print('中继%s(%s): 海拔%.0f 离地%.0f 到岗%.0f 离岗%.0f 准备%.0f 周转完%.0f 充电完%.0f 服务区%s'
          % (rid, r['中继编号'], r['海拔m'], r['离地高度m'], r['到岗时刻s'], r['离岗时刻s'],
             prep, back, chg, ','.join(relay_sas_of.get(rid, []))))

# ---------- 完整区间资源核算 ----------
def group_resources(sa_set):
    sa_set = set(sa_set)
    sel_t = [t for t in task_list if all(s in sa_set for s in t['sas'])]
    rel_in = [rl for rl in relay_list if set(rl['sas']) & sa_set]
    cross = [rl for rl in rel_in if not (set(rl['sas']) <= sa_set)]
    need = {}
    for mt in ['A', 'B', 'C']:
        need['uav' + mt] = peak_concurrent([(t['prep'], t['ret']) for t in sel_t if t['mtype'] == mt])
        need['bat' + mt] = peak_concurrent([(t['prep'], t['bat_full']) for t in sel_t if t['mtype'] == mt])
    need['relay'] = peak_concurrent([(rl['prep'], rl['back']) for rl in rel_in])
    need['relay_comp'] = peak_concurrent([(rl['prep'], rl['chg_done']) for rl in rel_in])
    return need, sel_t, rel_in, cross

g_need, g_t, g_r, _ = group_resources(set(s for t in task_list for s in t['sas']))
print('\n全局基准(完整区间):', dict(g_need), 'D=%d' % sum(g_need.values()))
print('中继机并发峰值=%d (库存2), 中继组件峰值=%d (库存6)' % (g_need['relay'], g_need['relay_comp']))

# ---------- 候选 ----------
from sklearn.cluster import KMeans
X = np.array([[sa_info[s]['lon'], sa_info[s]['lat']] for s in ALL_SA])
def bind_fix(lab):
    i11 = ALL_SA.index('S011'); i08 = ALL_SA.index('S008')
    lab = list(lab); lab[i08] = lab[i11]
    groups = defaultdict(list)
    for s, l in zip(ALL_SA, lab):
        groups[int(l)].append(s)
    return tuple(sorted(tuple(sorted(v)) for v in groups.values()))
def mk(parts):
    return frozenset(tuple(sorted(g)) for g in parts)
def valid(parts):
    ps = [set(g) for g in parts]
    if len(ps) != len(parts): return False
    u = set().union(*ps)
    if u != set(ALL_SA): return False
    if sum(len(g) for g in ps) != len(ALL_SA): return False
    for g in ps:
        if not g: return False
    for g in ps:
        if ('S011' in g) != ('S008' in g): return False
    return True

cand = {2: set(), 3: set()}
for K in [2, 3]:
    for seed in range(1, 81):
        km = KMeans(n_clusters=K, init='k-means++', n_init=1, random_state=seed, max_iter=300).fit(X)
        p = bind_fix(km.labels_)
        if valid(p): cand[K].add(frozenset(p))
skel = {
 3: [[['S002','S006','S007','S008','S010','S014'], ['S003','S004','S012','S013','S015'], ['S001','S005','S009','S011']]],
 2: [[['S002','S003','S004','S007','S008','S010','S012','S013','S014','S015'], ['S001','S005','S006','S009','S011']]]}
for K, plist in skel.items():
    for p in plist:
        pp = bind_fix([next(gi for gi, g in enumerate(p) if s in g) for s in ALL_SA])
        if valid(pp): cand[K].add(frozenset(pp))
relay_sas = sorted(set(s for rl in relay_list for s in rl['sas']))
z1 = [s for s in ALL_SA if s in relay_sas]; z2 = [s for s in ALL_SA if s not in relay_sas]
cand[2].add(mk([z1, z2])); cand[3].add(mk([z1, ['S001'], ['S006']]))
print('候选数: K=2 -> %d, K=3 -> %d' % (len(cand[2]), len(cand[3])))

# ---------- 评价 ----------
def balance_index(parts):
    stat = []
    for grp in parts:
        tsel = [t for t in task_list if all(s in grp for s in t['sas'])]
        boxes = sum(sa_demand[s]['boxes'] for s in grp)
        mass = sum(sa_demand[s]['mass'] for s in grp)
        n = len(tsel); e = sum(t['energy'] for t in tsel)
        dur = max((t['ret'] for t in tsel), default=0) - min((t['prep'] for t in tsel), default=0)
        rly = len([rl for rl in relay_list if set(rl['sas']) & set(grp)])
        stat.append([boxes, mass, n, e, dur, rly])
    stat = np.array(stat, dtype=float)
    cv = np.std(stat, axis=0) / np.maximum(np.mean(stat, axis=0), 1e-9)
    return float(np.mean(cv))

def eval_partition(parts, do_balance=True):
    need_total = defaultdict(int); per = []
    rid_groups = defaultdict(set)
    for grp in parts:
        need, _, _, cross = group_resources(grp)
        for k in need: need_total[k] += need[k]
        per.append(len(cross))
        for rl in cross: rid_groups[rl['rid']].add(grp.__repr__())
    gap = {k: max(0, need_total[k] - INV[k]) for k in need_total}
    redun = {k: max(0, need_total[k] - g_need[k]) for k in need_total}
    D = sum(need_total.values()); G = sum(gap.values())
    cross_u = len(rid_groups)
    cross_extra = sum(len(v) - 1 for v in rid_groups.values())
    J = balance_index(parts) if do_balance else float('nan')
    return dict(need=dict(need_total), gap=gap, redun=redun, D=D, G=G, J=J,
                cross=per, cross_t=sum(per), cross_u=cross_u, cross_extra=cross_extra)

def min_group_ok(parts, min_sa=3):
    return all(len(set(g)) >= min_sa for g in parts)

results = {}
for K in [2, 3]:
    cands = sorted(cand[K], key=lambda c: sorted([len(g) for g in c]))
    best = None
    for cp in cands:
        parts = tuple(sorted([tuple(sorted(g)) for g in cp]))
        if not min_group_ok(parts):
            continue
        r = eval_partition(parts)
        key = (r['cross_u'], r['cross_extra'], r['D'], r['G'], r['J'])
        if best is None or key < best[0]:
            best = (key, parts, r)
    results[K] = best
    print('\n=== K=%d 最优（均衡约束下）===' % K)
    for grp in best[1]:
        print('  ', grp)
    print('  跨组架次数=%d 复制总数=%d 每组分%s' % (best[2]['cross_u'], best[2]['cross_extra'], best[2]['cross']))
    print('  需求:', best[2]['need'])
    print('  缺口:', best[2]['gap'], ' 冗余:', best[2]['redun'])
    print('  D=%d G=%d J=%.4f' % (best[2]['D'], best[2]['G'], best[2]['J']))

for K, parts in [(2, [z1, z2]), (3, [z1, ['S001'], ['S006']])]:
    r = eval_partition(parts)
    print('\n零复制对照 K=%d: D=%d G=%d 跨组架次=%d 复制=%d' % (K, r['D'], r['G'], r['cross_u'], r['cross_extra']))

# ---------- 落盘 ----------
def group_detail(parts):
    rows = []
    for gi, grp in enumerate(parts):
        tsel = [t for t in task_list if all(s in grp for s in t['sas'])]
        rel_in = [rl for rl in relay_list if set(rl['sas']) & set(grp)]
        cross = [rl['rid'] for rl in rel_in if not (set(rl['sas']) <= set(grp))]
        whole = [rl['rid'] for rl in rel_in if set(rl['sas']) <= set(grp)]
        rows.append(dict(
            group=gi+1, sas=sorted(grp),
            boxes=sum(sa_demand[s]['boxes'] for s in grp),
            mass=round(sum(sa_demand[s]['mass'] for s in grp), 1),
            vol=round(sum(sa_demand[s]['vol'] for s in grp), 3),
            first=sum(sa_demand[s]['first'] for s in grp),
            avg_dist=round(np.mean([hav(O_LON, O_LAT, sa_info[s]['lon'], sa_info[s]['lat']) for s in grp]), 1),
            trips=[t['tid'] for t in tsel], n_trips=len(tsel),
            energy=round(sum(t['energy'] for t in tsel), 4),
            relay_whole=whole, relay_cross=cross,
            need={k: v for k, v in group_resources(grp)[0].items()},
        ))
    return rows

out = dict(inventory=INV, global_need=g_need,
    global_detail=group_detail([ALL_SA]),
    K2=dict(parts=[list(p) for p in results[2][1]], detail=group_detail(results[2][1]), **results[2][2]),
    K3=dict(parts=[list(p) for p in results[3][1]], detail=group_detail(results[3][1]), **results[3][2]),
    zero_cross={'K2': eval_partition([z1, z2], do_balance=False), 'K3': eval_partition([z1, ['S001'], ['S006']], do_balance=False)})
with open(BASE + r'\Q4_分区核算结果_v3.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1, default=str)
print('\n已保存 Q4_分区核算结果_v3.json')
