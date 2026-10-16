# -*- coding: utf-8 -*-
"""
Problem 2 external baseline: OR-Tools CP-SAT exact scheduling vs urgency-first greedy.

Fair comparison protocol:
  - SAME fixed task set: the 20 trips of the recommended plan
    (box grouping and zone visit order frozen, read from frozen CSVs).
  - SAME physics: per-leg DEM cruise altitude, nonlinear equivalent range Lg(q),
    climb surcharge, two-stage charging, rho=20% reserve (identical to Q2_精确调度.py).
  - Each task may fly any physically feasible UAV type in {A,B,C}; the solver
    chooses type, numbered aircraft, battery and prep-start time.
  - Greedy = deadline-ordered earliest feasible resource (the paper's scheduler).
  - CP-SAT lex-minimizes (hard-deadline violations, Cmax, energy, soft tardiness),
    time limit reported below; prints proven optimality gap and wall time.

Outputs: q2_cpsat_real.json / q2_cpsat_real.csv
"""
import os, json, time
import numpy as np
import pandas as pd
from scipy.io import loadmat
from math import radians, sin, cos, sqrt, atan2
from ortools.sat.python import cp_model

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = BASE + r'\数据'
PREP = BASE + r'\预处理输出'
OUT = os.path.dirname(os.path.abspath(__file__))
DEM_MAT = DATA + r'\镇龙乡地理空间数据\镇龙乡及周边地理数据\数字高程模型数据（DEM）\镇龙乡及周边30米DEM.mat'
G = 9.81
TIME_LIMIT_S = 300.0
SEED = 20251015
NUM_WORKERS = 8

# ---------------- physics (same calibre as Q2_精确调度.py) ----------------
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

h_op = {r['编号']: float(r['海拔(m)']) + 30.0 for _, r in sa_rows.iterrows()}
h_op['O01'] = O_ALT
lon = {r['编号']: float(r['经度']) for _, r in sa_rows.iterrows()}
lat = {r['编号']: float(r['纬度']) for _, r in sa_rows.iterrows()}
lon['O01'], lat['O01'] = O_LON, O_LAT

_zones = sa_rows['编号'].tolist() + ['O01']
Hc = {}
for i, a in enumerate(_zones):
    for b in _zones[i:]:
        Hc[(a, b)] = Hc[(b, a)] = cruise_exact(lon[a], lat[a], lon[b], lat[b])

def seg_time(pp, d, Hc_val, h0, h1):
    return max(Hc_val - h0, 0.0) / pp['v_up'] + d / pp['v'] + max(Hc_val - h1, 0.0) / pp['v_dn']

def seg_energy(pp, q, d, Hc_val, h0, h1):
    return pp['E'] * d / Lg(pp, q) + 9.81 * (pp['m0'] + q) * max(Hc_val - h0, 0.0) / (pp['eta'] * 3600000)

box_w = {b['货箱序号']: float(b['单箱质量(kg)']) for _, b in cargo.iterrows()}
box_v = {b['货箱序号']: float(b['单箱体积(m³)']) for _, b in cargo.iterrows()}

def task_option(seq, boxes_by_zone, mt):
    """Identical computation to Q2_精确调度.task_option for a given ordered zone tour."""
    pp = pps[mt]
    total_w = sum(box_w[b] for z in seq[1:-1] for b in boxes_by_zone[z])
    if total_w > pp['Q'] + 1e-9:
        return None
    vol = sum(box_v[b] for z in seq[1:-1] for b in boxes_by_zone[z])
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
            rem -= sum(box_w[bx] for bx in boxes_by_zone[b])
    if E > (1 - pp['rho']) * pp['E'] + 1e-9:
        return None
    T = sum(seg_time(pp, hav(lon[seq[k]], lat[seq[k]], lon[seq[k+1]], lat[seq[k+1]]),
                     Hc[(seq[k], seq[k+1])], h_op[seq[k]], h_op[seq[k+1]])
            for k in range(len(seq) - 1))
    n_total = sum(len(v) for v in boxes_by_zone.values())
    T += pp['t_prep'] + pp['t_load'] * n_total
    rel_arr = {}
    t_acc = pp['t_prep'] + pp['t_load'] * n_total
    for k in range(1, len(seq) - 1):
        t_acc += seg_time(pp, hav(lon[seq[k-1]], lat[seq[k-1]], lon[seq[k]], lat[seq[k]]),
                          Hc[(seq[k-1], seq[k])], h_op[seq[k-1]], h_op[seq[k]])
        rel_arr[seq[k]] = t_acc + pp['t_hand0'] + pp['t_hand1'] * len(boxes_by_zone[seq[k]])
        t_acc += pp['t_hand0'] + pp['t_hand1'] * len(boxes_by_zone[seq[k]])
    boxes = [b for z in seq[1:-1] for b in boxes_by_zone[z]]
    return dict(mt=mt, E=E, T=T, rel_arr=rel_arr, boxes=boxes, n=len(boxes), total_w=total_w)

def charge_time(s, Tfull):
    if s < 0.9:
        return Tfull * (0.65 * (0.9 - s) / 0.9 + 0.35)
    return Tfull * 0.35 * (1.0 - s) / 0.1

# ---------------- fixed task set from frozen recommended plan ----------------
trips = pd.read_csv(BASE + r'\Q2_主方案20架次_架次表.csv')
boxes_df = pd.read_csv(BASE + r'\Q2_主方案20架次_逐箱交付.csv')

hard_deadline = {}
soft_exp = {}
box_zone = {}
for _, b in boxes_df.iterrows():
    bid = b['货箱序号']
    box_zone[bid] = b['服务区']
    if not pd.isna(b['硬截止s']):
        hard_deadline[bid] = float(b['硬截止s'])
    soft_exp[bid] = float(b['期望送达s'])

tasks = []
for _, r in trips.iterrows():
    tid = r['架次ID']
    seq = [p.strip() for p in str(r['路径']).split('→')]
    tboxes = boxes_df[boxes_df['架次ID'] == tid]['货箱序号'].tolist()
    bz = {z: [b for b in tboxes if box_zone[b] == z] for z in seq[1:-1]}
    opts = []
    for mt in ['A', 'B', 'C']:
        op = task_option(seq, bz, mt)
        if op is not None:
            op['soc'] = 1.0 - op['E'] / pps[mt]['E']
            op['tchg'] = charge_time(op['soc'], TCHG[mt])
            opts.append(op)
    dl = min((hard_deadline[b] for b in tboxes if b in hard_deadline), default=float('inf'))
    tasks.append(dict(tid=tid, seq=seq, boxes=tboxes, bz=bz, options=opts, deadline=dl))
    assert opts, f'{tid} has no feasible UAV type'
print(f'tasks={len(tasks)}, feasible task-type pairs={sum(len(t["options"]) for t in tasks)}')

# ---------------- greedy (paper scheduler, same task set) ----------------
def schedule_greedy(task_list):
    uav_free = {f'{mt}{i}': 0.0 for mt in FLEET for i in range(1, FLEET[mt] + 1)}
    bat_free = {mt: [0.0] * BATS[mt] for mt in FLEET}
    remain = sorted(task_list, key=lambda t: t['deadline'])
    sched = []
    for pick in remain:
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
        bat_free[mt][bat_idx] = t_ret + opt['tchg']
        uav_free[uav_id] = t_ret
        arr = {z: t_dep + opt['rel_arr'][z] for z in opt['rel_arr']}
        sched.append(dict(tid=pick['tid'], mt=mt, dep=t_dep, ret=t_ret, E=opt['E'],
                          arr=arr, boxes=opt['boxes'], seq=pick['seq']))
    Cmax = max(s['ret'] for s in sched)
    E = sum(s['E'] for s in sched)
    hard_late = 0
    soft_min = 0.0
    for s in sched:
        for b in s['boxes']:
            z = box_zone[b]
            if b in hard_deadline and s['arr'][z] > hard_deadline[b] + 1e-9:
                hard_late += 1
            if b not in hard_deadline and s['arr'][z] > soft_exp[b] + 1e-9:
                soft_min += (s['arr'][z] - soft_exp[b]) / 60.0
    return dict(Cmax=Cmax, E=E, hard_late=hard_late, soft_min=soft_min, sched=sched)

t0 = time.perf_counter()
greedy = schedule_greedy(tasks)
t_greedy = time.perf_counter() - t0
print(f'GREEDY  Cmax={greedy["Cmax"]/3600:.4f} h  E={greedy["E"]:.4f} kWh  '
      f'hard-late={greedy["hard_late"]}  soft={greedy["soft_min"]:.2f} min  time={t_greedy*1000:.1f} ms')

# ---------------- CP-SAT exact scheduling ----------------
HORIZON = 20000
W_CMAX, W_ENERGY, W_SOFT = 10**8, 10, 1

mdl = cp_model.CpModel()
aircraft = [f'{mt}{i}' for mt in FLEET for i in range(1, FLEET[mt] + 1)]
batteries = [(mt, k) for mt in FLEET for k in range(BATS[mt])]
ac_intervals = {ac: [] for ac in aircraft}
bat_intervals = {bk: [] for bk in batteries}

starts, ends, pres, hard_viol = [], [], [], []
ac_var, bat_var = {}, {}
cmax = mdl.NewIntVar(0, HORIZON, 'Cmax')

for ti, t in enumerate(tasks):
    s = mdl.NewIntVar(0, HORIZON, f's{ti}')
    e = mdl.NewIntVar(0, HORIZON, f'e{ti}')
    starts.append(s); ends.append(e)
    b_to = [mdl.NewBoolVar(f'b_{ti}_{oi}') for oi in range(len(t['options']))]
    pres.append(b_to)
    mdl.AddExactlyOne(b_to)
    for oi, op in enumerate(t['options']):
        mdl.Add(e == s + int(np.ceil(op['T']))).OnlyEnforceIf(b_to[oi])
    mdl.Add(cmax >= e)

    for oi, op in enumerate(t['options']):
        mt = op['mt']
        dur = int(np.ceil(op['T']))
        chg = int(np.ceil(op['tchg']))
        # aircraft of matching type: exactly one if option chosen
        ac_sel = []
        for ac in aircraft:
            if not ac.startswith(mt):
                continue
            xa = mdl.NewBoolVar(f'x_{ti}_{oi}_{ac}')
            mdl.Add(xa <= b_to[oi])
            iv = mdl.NewOptionalIntervalVar(s, dur, e, xa, f'aciv_{ti}_{ac}')
            ac_intervals[ac].append(iv)
            ac_sel.append(xa)
            ac_var[(ti, oi, ac)] = xa
        mdl.Add(sum(ac_sel) == 1).OnlyEnforceIf(b_to[oi])
        # battery of matching type: occupied [prep-start, recharge-complete]
        bat_sel = []
        for bk in batteries:
            if bk[0] != mt:
                continue
            yb = mdl.NewBoolVar(f'y_{ti}_{oi}_{bk[0]}{bk[1]}')
            mdl.Add(yb <= b_to[oi])
            ivb = mdl.NewOptionalIntervalVar(s, dur + chg, s + dur + chg, yb,
                                             f'bativ_{ti}_{bk[0]}{bk[1]}')
            bat_intervals[bk].append(ivb)
            bat_sel.append(yb)
            bat_var[(ti, oi, bk)] = yb
        mdl.Add(sum(bat_sel) == 1).OnlyEnforceIf(b_to[oi])

    # hard deadlines (tightest box deadline per zone)
    for oi, op in enumerate(t['options']):
        for z in t['bz']:
            hb = [b for b in t['bz'][z] if b in hard_deadline]
            if not hb:
                continue
            dl = int(min(hard_deadline[b] for b in hb))
            offs = int(np.ceil(op['rel_arr'][z]))
            late = mdl.NewBoolVar(f'late_{ti}_{oi}_{z}')
            mdl.Add(s + offs > dl).OnlyEnforceIf(late)
            mdl.Add(s + offs <= dl).OnlyEnforceIf(late.Not())
            mdl.Add(late <= b_to[oi])
            hard_viol.append(late)

mdl.Add(sum(hard_viol) == 0)

energy_terms = [int(round(op['E'] * 1000)) * pres[ti][oi]
                for ti, t in enumerate(tasks) for oi, op in enumerate(t['options'])]
soft_vars = []
for ti, t in enumerate(tasks):
    for b in t['boxes']:
        if b in hard_deadline:
            continue
        z = box_zone[b]
        exp = int(round(soft_exp[b]))
        d = mdl.NewIntVar(0, HORIZON, f'd_{ti}_{b}')
        for oi, op in enumerate(t['options']):
            offs = int(np.ceil(op['rel_arr'][z]))
            mdl.Add(d >= starts[ti] + offs - exp).OnlyEnforceIf(pres[ti][oi])
        soft_vars.append(d)

mdl.Minimize(W_CMAX * cmax + W_ENERGY * sum(energy_terms) + W_SOFT * sum(soft_vars))

for ac in aircraft:
    mdl.AddNoOverlap(ac_intervals[ac])
for bk in batteries:
    mdl.AddNoOverlap(bat_intervals[bk])

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = TIME_LIMIT_S
solver.parameters.num_search_workers = NUM_WORKERS
solver.parameters.random_seed = SEED
t0 = time.perf_counter()
status = solver.Solve(mdl)
t_cpsat = time.perf_counter() - t0
assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), f'no feasible schedule, status={status}'

# extract solution
E_wh = sum(int(round(op['E'] * 1000)) * solver.Value(pres[ti][oi])
           for ti, t in enumerate(tasks) for oi, op in enumerate(t['options']))
soft_s = sum(solver.Value(v) for v in soft_vars)
cpsat = dict(Cmax=float(solver.Value(cmax)), E=E_wh / 1000.0, hard_late=0,
             soft_min=soft_s / 60.0)
print(f'CP-SAT  Cmax={cpsat["Cmax"]/3600:.4f} h  E={cpsat["E"]:.4f} kWh  '
      f'hard-late=0  soft={cpsat["soft_min"]:.2f} min  time={t_cpsat:.2f} s  '
      f'status={"OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"}  gap={solver.BestObjectiveBound()*0 + (solver.ObjectiveValue()-solver.BestObjectiveBound())/max(1,abs(solver.ObjectiveValue()))*100:.3f}%')

# ---------- extract assignment + independent Python verification ----------
# All CP durations are ceil() of continuous values, so integer CP windows
# dominate continuous physical windows: a CP-feasible plan is strictly feasible
# under continuous physics (zero rounding slack needed).
TOL = 1e-6
assign = []
ac_windows = {ac: [] for ac in aircraft}
bat_windows = {bk: [] for bk in batteries}
verify_errors = []
E_sum = 0.0
hard_late_cont = 0
soft_min_cont = 0.0

for ti, t in enumerate(tasks):
    oi = next(k for k, bv in enumerate(pres[ti]) if solver.Value(bv))
    op = t['options'][oi]
    mt = op['mt']
    s0 = float(solver.Value(starts[ti]))
    chosen_ac = next(k[2] for k, v in ac_var.items()
                     if k[0] == ti and k[1] == oi and solver.Value(v))
    chosen_bat = next(k[2] for k, v in bat_var.items()
                      if k[0] == ti and k[1] == oi and solver.Value(v))
    # continuous physics values
    ret_c = s0 + op['T']
    rel_c = {z: s0 + op['rel_arr'][z] for z in op['rel_arr']}
    assign.append(dict(tid=t['tid'], mt=mt, aircraft=chosen_ac,
                       battery=f'{chosen_bat[0]}{chosen_bat[1]+1}',
                       start=round(s0, 1), return_s=round(ret_c, 1),
                       E=round(op['E'], 4), soc=round(op['soc'], 4)))
    ac_windows[chosen_ac].append((s0, ret_c, t['tid']))
    bat_windows[chosen_bat].append((s0, s0 + op['T'] + op['tchg'], t['tid']))
    E_sum += op['E']
    for b in t['boxes']:
        z = box_zone[b]
        arr = rel_c[z]
        if b in hard_deadline:
            if arr > hard_deadline[b] + TOL:
                hard_late_cont += 1
                verify_errors.append(f'{b} hard deadline {hard_deadline[b]:.0f} missed arr={arr:.1f}')
        elif arr > soft_exp[b] + TOL:
            soft_min_cont += (arr - soft_exp[b]) / 60.0

# aircraft non-overlap
for ac, wins in ac_windows.items():
    wins.sort()
    for (s1, e1, t1), (s2, e2, t2) in zip(wins, wins[1:]):
        if s2 < e1 - TOL:
            verify_errors.append(f'aircraft {ac} overlap {t1}/{t2}: {e1:.1f} > {s2:.1f}')
# battery non-overlap (through recharge completion)
for bk, wins in bat_windows.items():
    wins.sort()
    for (s1, e1, t1), (s2, e2, t2) in zip(wins, wins[1:]):
        if s2 < e1 - TOL:
            verify_errors.append(f'battery {bk} overlap {t1}/{t2}: ready {e1:.1f} > start {s2:.1f}')

cmax_cont = max(a['return_s'] for a in assign)
print(f'VERIFY  continuous: Cmax={cmax_cont/3600:.4f} h  E={E_sum:.4f} kWh  '
      f'hard-late={hard_late_cont}  soft={soft_min_cont:.2f} min  errors={len(verify_errors)}')
if verify_errors:
    for e in verify_errors[:10]:
        print('  ERROR:', e)
    raise SystemExit('CP-SAT solution failed independent verification')

result = dict(
    instance='real_20trip', time_limit_s=TIME_LIMIT_S, seed=SEED,
    greedy=dict(Cmax_s=round(greedy['Cmax'], 2), Cmax_h=round(greedy['Cmax'] / 3600, 4),
                E_kWh=round(greedy['E'], 4), hard_late=greedy['hard_late'],
                soft_min=round(greedy['soft_min'], 2), wall_s=round(t_greedy, 4)),
    cpsat=dict(Cmax_s=round(cpsat['Cmax'], 2), Cmax_h=round(cpsat['Cmax'] / 3600, 4),
               E_kWh=round(cpsat['E'], 4), hard_late=0,
               soft_min=round(cpsat['soft_min'], 2), wall_s=round(t_cpsat, 3),
               optimal=status == cp_model.OPTIMAL,
               objective=float(solver.ObjectiveValue()),
               bound=float(solver.BestObjectiveBound()),
               gap_pct=round(float((solver.ObjectiveValue() - solver.BestObjectiveBound())
                                   / max(1.0, abs(solver.ObjectiveValue()))) * 100, 4)),
    independent_verify=dict(passed=not verify_errors, Cmax_h=round(cmax_cont / 3600, 4),
                            E_kWh=round(E_sum, 4), hard_late=hard_late_cont,
                            soft_min=round(soft_min_cont, 2),
                            cp_vs_continuous_Cmax_diff_s=round(cpsat['Cmax'] - cmax_cont, 3)),
    cpsat_assignment=assign)
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'q2_cpsat_real.json'), 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=1)
pd.DataFrame([dict(method='greedy', **{k: v for k, v in result['greedy'].items()}),
              dict(method='cp-sat', **{k: v for k, v in result['cpsat'].items() if k != 'cpsat_assignment'})]
             ).to_csv(os.path.join(OUT, 'q2_cpsat_real.csv'), index=False, encoding='utf-8-sig')
print('saved q2_cpsat_real.json/csv')

# empirical option/deadline pool for the synthetic robustness family
pool = {'options_by_mt': {mt: [] for mt in ['A', 'B', 'C']},
        'hard_deadlines': [], 'n_tasks': len(tasks)}
for t in tasks:
    feasible_mts = [o['mt'] for o in t['options']]
    for o in t['options']:
        pool['options_by_mt'][o['mt']].append(dict(T=round(o['T'], 2), E=round(o['E'], 4),
                                                   tchg=round(o['tchg'], 2)))
    if t['deadline'] < float('inf'):
        pool['hard_deadlines'].append(int(t['deadline']))
# feasible-type pattern frequency (bitmask A=1,B=2,C=4 over tasks)
from collections import Counter
pat = Counter()
for t in tasks:
    mask = sum({'A': 1, 'B': 2, 'C': 4}[o['mt']] for o in t['options'])
    pat[mask] += 1
pool['pattern_freq'] = {str(k): v for k, v in pat.items()}
with open(os.path.join(OUT, 'q2_empirical_pool.json'), 'w', encoding='utf-8') as f:
    json.dump(pool, f, ensure_ascii=False, indent=1)
print('saved q2_empirical_pool.json  patterns=', dict(pat))
