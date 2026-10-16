# -*- coding: utf-8 -*-
"""
Problem 4 external baselines: K-means / agglomerative (Ward) clustering vs SC-MAE-RG.

Fair comparison protocol:
  - SAME 15 service areas, SAME frozen transport/relay timeline,
    SAME inventory limits, SAME evaluator (peak-concurrency resource accounting,
    gap G, duplication, balance index J) as Q4_分区核算_v3.py.
  - Feature sets:
      geo  = standardized (lon, lat)
      full = standardized (lon, lat, boxes, mass, vol, first)
  - Methods: K-means (n_init=100, seed 20251015), Ward agglomerative.
  - All outputs receive the identical feasibility repair (S008 forced into the
    S011 group because T18 serves both) and the >=3-area group-size rule used by
    the paper's candidate enumeration.
  - SC-MAE-RG numbers are read from the frozen v3 result JSON.

Outputs: q4_cluster_baseline.json / q4_cluster_baseline.csv
"""
import os, json
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREP = BASE + r'\预处理输出'
OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 20251015

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

# ---------- service areas + demand ----------
nodes = pd.read_csv(PREP + r'\Q1_节点_预处理.csv')
o = nodes[nodes['编号'] == 'O01'].iloc[0]
O_LON, O_LAT = float(o['经度']), float(o['纬度'])
sa_rows = nodes[nodes['编号'] != 'O01'].copy().reset_index(drop=True)
sa_info = {r['编号']: dict(lon=float(r['经度']), lat=float(r['纬度']), elev=float(r['海拔(m)']))
           for _, r in sa_rows.iterrows()}
ALL_SA = sorted(sa_info.keys())
_q4old = json.load(open(BASE + r'\Q4_分区核算结果.json', encoding='utf-8'))
sa_demand = defaultdict(lambda: dict(boxes=0, mass=0, vol=0, first=0))
for row in _q4old['sa_table']:
    sa_demand[row['sa']] = dict(boxes=int(row['boxes']), mass=float(row['mass']),
                                vol=float(row.get('vol', 0.0)), first=int(row.get('first', 0)))

# ---------- transport tasks (frozen 20-trip plan) ----------
def sa_of_path(path):
    return [p for p in str(path).split('→') if p not in ('O01', '') and p.strip()]

trips = pd.read_csv(BASE + r'\Q2_主方案20架次_架次表.csv')
sched = pd.read_csv(BASE + r'\Q2_主方案20架次_无人机时刻表.csv')
trip_meta = {r['架次ID']: dict(battery=r['电池'], bat_full=float(r['电池充满s']),
                               prep=float(r['准备开始s']), takeoff=float(r['起飞s']),
                               ret=float(r['返航s'])) for _, r in sched.iterrows()}
task_list = []
for _, r in trips.iterrows():
    tid = r['架次ID']; mt = trip_meta[tid]
    task_list.append(dict(tid=tid, uav=r['无人机'], mtype=r['机型'],
                          sas=sa_of_path(r['路径']), prep=mt['prep'], takeoff=mt['takeoff'],
                          ret=mt['ret'], battery=mt['battery'], bat_full=mt['bat_full'],
                          energy=float(r['能耗kWh'])))

# ---------- relay tasks (frozen Q3 timeline, objects derived from per-point log) ----------
relays = pd.read_csv(BASE + r'\Q3_两架方案_中继排程.csv')
perpt = pd.read_csv(BASE + r'\Q3_两架方案_逐点通信验证.csv')
relay_sas_of = defaultdict(set)
for _, p in perpt.iterrows():
    w = p['保障中继']
    if not isinstance(w, str) or not w:
        continue
    tid_task = w.split('-', 1)[1] if '-' in w else w
    for t in task_list:
        if t['tid'] == p['架次']:
            relay_sas_of[tid_task].update(t['sas'])
relay_list = []
for _, r in relays.iterrows():
    rid = str(r['任务ID'])
    relay_list.append(dict(rid=rid, machine=str(r['中继编号']),
                           sas=sorted(relay_sas_of.get(rid, [])),
                           prep=float(r['准备开始s']), back=float(r['周转完成s']),
                           chg_done=float(r['充电完成s'])))

# ---------- evaluator (identical to Q4_分区核算_v3.py) ----------
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

g_need, _, _, _ = group_resources(set(ALL_SA))

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
    need_total = defaultdict(int)
    rid_groups = defaultdict(set)
    per = []
    for grp in parts:
        need, _, _, cross = group_resources(grp)
        for k in need:
            need_total[k] += need[k]
        per.append(len(cross))
        for rl in cross:
            rid_groups[rl['rid']].add(tuple(sorted(grp)))
    gap = {k: max(0, need_total[k] - INV[k]) for k in need_total}
    D = sum(need_total.values()); G = sum(gap.values())
    cross_u = len(rid_groups)
    cross_extra = sum(len(v) - 1 for v in rid_groups.values())
    J = balance_index(parts) if do_balance else float('nan')
    return dict(D=int(D), G=int(G), J=round(J, 4), cross_u=cross_u, cross_extra=cross_extra,
                need={k: int(v) for k, v in need_total.items()})

def bind_only(labels):
    """Apply the one hard coupling every partition must respect: S008 shares
    trip T18 with S011, so both areas must belong to the same group.
    Returns the partition (groups may be small; min-size is reported separately)."""
    lab = list(labels)
    i11, i08 = ALL_SA.index('S011'), ALL_SA.index('S008')
    lab[i08] = lab[i11]
    groups = defaultdict(list)
    for s, l in zip(ALL_SA, lab):
        groups[int(l)].append(s)
    return tuple(sorted(tuple(sorted(v)) for v in groups.values()))

# ---------- feature matrices ----------
X_geo_raw = np.array([[sa_info[s]['lon'], sa_info[s]['lat']] for s in ALL_SA])
X_full_raw = np.array([[sa_info[s]['lon'], sa_info[s]['lat'],
                        sa_demand[s]['boxes'], sa_demand[s]['mass'],
                        sa_demand[s]['vol'], sa_demand[s]['first']] for s in ALL_SA])
X_geo = StandardScaler().fit_transform(X_geo_raw)
X_full = StandardScaler().fit_transform(X_full_raw)

MIN_GROUP = 3

def run_method(name, K, labels):
    parts = bind_only(labels)
    r = eval_partition(parts)
    sizes = sorted(len(g) for g in parts)
    return dict(method=name, K=K, feasible=True, parts=[list(g) for g in parts],
                group_sizes=sizes, min_size_ok=all(n >= MIN_GROUP for n in sizes), **r)

rows = []
for K in (2, 3):
    km_geo = KMeans(n_clusters=K, init='k-means++', n_init=100, random_state=SEED, max_iter=1000)
    rows.append(run_method('K-means (geo)', K, km_geo.fit_predict(X_geo)))
    km_full = KMeans(n_clusters=K, init='k-means++', n_init=100, random_state=SEED, max_iter=1000)
    rows.append(run_method('K-means (geo+demand)', K, km_full.fit_predict(X_full)))
    rows.append(run_method('Ward (geo)', K, AgglomerativeClustering(n_clusters=K, linkage='ward').fit_predict(X_geo)))
    rows.append(run_method('Ward (geo+demand)', K,
                           AgglomerativeClustering(n_clusters=K, linkage='ward').fit_predict(X_full)))

# ---------- paper method (frozen v3 results) ----------
v3 = json.load(open(BASE + r'\Q4_分区核算结果_v3.json', encoding='utf-8'))
paper = {}
for K, key in ((2, 'K2'), (3, 'K3')):
    d = v3[key]
    parts = tuple(sorted(tuple(g) for g in d['parts']))
    r = eval_partition(parts)
    paper[K] = dict(method='SC-MAE-RG (proposed)', K=K, feasible=True,
                    parts=[list(g) for g in parts], group_sizes=sorted(len(g) for g in parts),
                    min_size_ok=all(len(g) >= MIN_GROUP for g in parts), **r)
    rows.append(paper[K])

print(f'{"method":24s} {"K":>1s}  {"D":>3s} {"G":>3s} {"J":>7s} {"crossU":>6s} {"dup":>3s}  sizes  >=3')
for r in rows:
    print(f'{r["method"]:24s} {r["K"]:>1d}  {r["D"]:>3d} {r["G"]:>3d} {r["J"]:>7.4f} '
          f'{r["cross_u"]:>6d} {r["cross_extra"]:>3d}  {str(r["group_sizes"]):7s} {"yes" if r["min_size_ok"] else "NO":>3s}')

out = dict(seed=SEED, global_need={k: int(v) for k, v in g_need.items()}, rows=rows,
           paper_D31_G5_J0613=dict(D=31, G=5, J=0.613, note='values quoted in manuscript'))
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'q4_cluster_baseline.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
pd.DataFrame([{k: v for k, v in r.items() if k != 'parts'} for r in rows]
             ).to_csv(os.path.join(OUT, 'q4_cluster_baseline.csv'), index=False, encoding='utf-8-sig')
print('saved q4_cluster_baseline.json/csv')
