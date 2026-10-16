# -*- coding: utf-8 -*-
"""
Problem 2 robustness family: 30 synthetic instances, greedy vs CP-SAT.

Instance generator is calibrated ONLY on empirical quantities exported from the
real 20-trip instance (q2_empirical_pool.json):
  - feasible-UAV-type pattern per task: C-only / B+C / A+B+C at observed 9:7:4,
  - per-type (duration T, energy E, recharge tchg) drawn with replacement from
    the 35 physically verified options,
  - tight task deadline drawn from the observed deadline multiset
    (10x3600 s, 5x7200 s, 3x10800 s) with the observed 18/20 tight-task share,
  - non-tight tasks carry a soft expectation of 7200 s.
Fleet/battery inventory, two-stage charging and lexicographic objective are
identical to q2_cpsat_baseline.py. Same ceil() conservative discretisation.

Outputs: q2_family_results.csv, q2_family_summary.json
(mean +/- std, median [IQR], paired Wilcoxon signed-rank, win/tie/loss)
"""
import os, json, time
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
from scipy import stats

import os
OUT = os.path.dirname(os.path.abspath(__file__))
SEED0 = 20251015
N_INS = 30
N_TASKS = 20
CP_TIME = 30.0
REL_OFFS_FRAC = 0.80          # delivery completion fraction of task duration (single-zone)

FLEET = {'A': 4, 'B': 2, 'C': 2}
BATS = {'A': 6, 'B': 4, 'C': 4}
aircraft = [f'{mt}{i}' for mt in FLEET for i in range(1, FLEET[mt] + 1)]
batteries = [(mt, k) for mt in FLEET for k in range(BATS[mt])]
HORIZON = 30000
W_CMAX, W_ENERGY, W_SOFT = 10**8, 10, 1

pool = json.load(open(os.path.join(OUT, 'q2_empirical_pool.json'), encoding='utf-8'))
OPTS = pool['options_by_mt']
PATTERNS = []
for mask, cnt in pool['pattern_freq'].items():
    m = int(mask)
    mts = [x for x, b in (('A', 1), ('B', 2), ('C', 4)) if m & b]
    PATTERNS += [mts] * cnt
DEADLINES = pool['hard_deadlines']                      # 18 values
TIGHT_SHARE = len(DEADLINES) / pool['n_tasks']          # 0.9

def make_instance(rng):
    inst = []
    for i in range(N_TASKS):
        mts = list(PATTERNS[rng.integers(len(PATTERNS))])
        options = []
        for mt in mts:
            row = OPTS[mt][rng.integers(len(OPTS[mt]))]
            options.append(dict(mt=mt, T=float(row['T']), E=float(row['E']),
                                tchg=float(row['tchg'])))
        tight = rng.random() < TIGHT_SHARE
        dl = float(DEADLINES[rng.integers(len(DEADLINES))]) if tight else float('inf')
        inst.append(dict(deadline=dl, options=options))
    return inst

def greedy(inst):
    uav_free = {f'{mt}{i}': 0.0 for mt in FLEET for i in range(1, FLEET[mt] + 1)}
    bat_free = {mt: [0.0] * BATS[mt] for mt in FLEET}
    t0 = time.perf_counter()
    rows = []
    soft = 0.0
    for ti, t in enumerate(sorted(inst, key=lambda x: x['deadline'])):
        best = None
        for op in t['options']:
            mt = op['mt']
            dep = max(min(uav_free[f'{mt}{i}'] for i in range(1, FLEET[mt] + 1)),
                      min(bat_free[mt]))
            if best is None or dep < best[0]:
                acid = min((f'{mt}{i}' for i in range(1, FLEET[mt] + 1)),
                           key=lambda k: uav_free[k])
                bk = min(range(BATS[mt]), key=lambda k: bat_free[mt][k])
                best = (dep, op, mt, acid, bk)
        dep, op, mt, acid, bk = best
        ret = dep + op['T']
        uav_free[acid] = ret
        bat_free[mt][bk] = ret + op['tchg']
        arr = dep + REL_OFFS_FRAC * op['T']
        late_h = np.isfinite(t['deadline']) and arr > t['deadline']
        if not np.isfinite(t['deadline']) and arr > 7200.0:
            soft += (arr - 7200.0) / 60.0
        rows.append((ret, op['E'], late_h))
    wall = time.perf_counter() - t0
    return dict(Cmax=max(r[0] for r in rows), E=sum(r[1] for r in rows),
                hard=int(sum(r[2] for r in rows)), soft=soft, wall=wall)

def cpsat(inst, seed):
    mdl = cp_model.CpModel()
    starts, pres, hard_viol, soft_vars = [], [], [], []
    ac_iv = {ac: [] for ac in aircraft}
    bat_iv = {bk: [] for bk in batteries}
    cmax = mdl.NewIntVar(0, HORIZON, 'cmax')
    for ti, t in enumerate(inst):
        s = mdl.NewIntVar(0, HORIZON, f's{ti}')
        e = mdl.NewIntVar(0, HORIZON, f'e{ti}')
        starts.append(s)
        bv = [mdl.NewBoolVar(f'b{ti}_{o}') for o in range(len(t['options']))]
        pres.append(bv)
        mdl.AddExactlyOne(bv)
        for oi, op in enumerate(t['options']):
            mdl.Add(e == s + int(np.ceil(op['T']))).OnlyEnforceIf(bv[oi])
        mdl.Add(cmax >= e)
        for oi, op in enumerate(t['options']):
            mt, dur, chg = op['mt'], int(np.ceil(op['T'])), int(np.ceil(op['tchg']))
            sel = []
            for ac in aircraft:
                if ac.startswith(mt):
                    x = mdl.NewBoolVar(f'x{ti}_{oi}_{ac}')
                    mdl.Add(x <= bv[oi])
                    ac_iv[ac].append(mdl.NewOptionalIntervalVar(s, dur, e, x, f'i{ti}_{ac}'))
                    sel.append(x)
            mdl.Add(sum(sel) == 1).OnlyEnforceIf(bv[oi])
            bsel = []
            for bk in batteries:
                if bk[0] == mt:
                    y = mdl.NewBoolVar(f'y{ti}_{oi}_{bk}')
                    mdl.Add(y <= bv[oi])
                    bat_iv[bk].append(mdl.NewOptionalIntervalVar(s, dur + chg, s + dur + chg, y, f'j{ti}_{bk}'))
                    bsel.append(y)
            mdl.Add(sum(bsel) == 1).OnlyEnforceIf(bv[oi])
            offs = int(np.ceil(REL_OFFS_FRAC * op['T']))
            if np.isfinite(t['deadline']):
                late = mdl.NewBoolVar(f'l{ti}_{oi}')
                mdl.Add(s + offs > int(t['deadline'])).OnlyEnforceIf(late)
                mdl.Add(s + offs <= int(t['deadline'])).OnlyEnforceIf(late.Not())
                mdl.Add(late <= bv[oi])
                hard_viol.append(late)
            else:
                d = mdl.NewIntVar(0, HORIZON, f'd{ti}')
                mdl.Add(d >= s + offs - 7200).OnlyEnforceIf(bv[oi])
                soft_vars.append(d)
    mdl.Add(sum(hard_viol) == 0)
    energy = [int(round(op['E'] * 1000)) * pres[ti][oi]
              for ti, t in enumerate(inst) for oi, op in enumerate(t['options'])]
    mdl.Minimize(W_CMAX * cmax + W_ENERGY * sum(energy) + W_SOFT * sum(soft_vars))
    for ac in aircraft:
        mdl.AddNoOverlap(ac_iv[ac])
    for bk in batteries:
        mdl.AddNoOverlap(bat_iv[bk])
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = CP_TIME
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = seed
    t0 = time.perf_counter()
    st = solver.Solve(mdl)
    wall = time.perf_counter() - t0
    assert st in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    E = sum(int(round(op['E'] * 1000)) * solver.Value(pres[ti][oi])
            for ti, t in enumerate(inst) for oi, op in enumerate(t['options'])) / 1000
    soft = sum(solver.Value(v) for v in soft_vars) / 60.0
    return dict(Cmax=float(solver.Value(cmax)), E=E, hard=0, soft=soft, wall=wall,
                optimal=st == cp_model.OPTIMAL)

rows = []
rej_total = 0
for k in range(N_INS):
    rng = np.random.default_rng(SEED0 + k)
    # rejection sampling: keep only instances feasible under hard deadlines
    # for BOTH schedulers (greedy feasibility certifies CP-SAT feasibility)
    inst, g, rej = None, None, 0
    while True:
        inst = make_instance(rng)
        g = greedy(inst)
        if g['hard'] == 0:
            break
        rej += 1
        assert rej < 1000, f'instance {k+1}: feasibility rejection runaway'
    rej_total += rej
    c = cpsat(inst, SEED0 + k)
    rows.append(dict(inst=k + 1,
                     g_Cmax_h=g['Cmax'] / 3600, c_Cmax_h=c['Cmax'] / 3600,
                     g_E=g['E'], c_E=c['E'],
                     g_hard=g['hard'], c_hard=c['hard'],
                     g_soft=g['soft'], c_soft=c['soft'],
                     g_wall_ms=g['wall'] * 1000, c_wall_s=c['wall'],
                     c_optimal=bool(c['optimal']),
                     improve_pct=(g['Cmax'] - c['Cmax']) / g['Cmax'] * 100))
    print(f'{k+1:2d}: greedy {g["Cmax"]/3600:.3f}h  cpsat {c["Cmax"]/3600:.3f}h  '
          f'imp {rows[-1]["improve_pct"]:5.1f}%  opt={c["optimal"]}  t={c["wall"]:.1f}s')

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'q2_family_results.csv'), index=False, encoding='utf-8-sig')
g, c = df['g_Cmax_h'], df['c_Cmax_h']
wil = stats.wilcoxon(c, g, alternative='less', zero_method='wilcox')
wins = int((g - c > 1e-9).sum()); ties = int((abs(g - c) <= 1e-9).sum()); losses = int((c - g > 1e-9).sum())
summary = dict(n=N_INS, rejected_infeasible=rej_total,
    greedy_Cmax=dict(mean=float(g.mean()), std=float(g.std(ddof=1)),
                     median=float(g.median()), q1=float(g.quantile(.25)), q3=float(g.quantile(.75)),
                     wall_ms_mean=float(df['g_wall_ms'].mean())),
    cpsat_Cmax=dict(mean=float(c.mean()), std=float(c.std(ddof=1)),
                    median=float(c.median()), q1=float(c.quantile(.25)), q3=float(c.quantile(.75)),
                    wall_s_mean=float(df['c_wall_s'].mean()),
                    wall_s_max=float(df['c_wall_s'].max()),
                    proven_optimal=int(df['c_optimal'].sum())),
    improve_pct_mean=float(df['improve_pct'].mean()),
    improve_pct_median=float(df['improve_pct'].median()),
    win_tie_loss=[wins, ties, losses],
    wilcoxon_W=float(wil.statistic), wilcoxon_p=float(wil.pvalue),
    energy_greedy_mean=float(df['g_E'].mean()), energy_cpsat_mean=float(df['c_E'].mean()),
    soft_greedy_mean=float(df['g_soft'].mean()), soft_cpsat_mean=float(df['c_soft'].mean()))
with open(os.path.join(OUT, 'q2_family_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)
print('\n=== SUMMARY ===')
print(f'greedy Cmax {g.mean():.3f}+-{g.std(ddof=1):.3f} h | cpsat {c.mean():.3f}+-{c.std(ddof=1):.3f} h')
print(f'median improve {summary["improve_pct_median"]:.1f}%, mean {summary["improve_pct_mean"]:.1f}%')
print(f'win/tie/loss {wins}/{ties}/{losses}, Wilcoxon p={wil.pvalue:.2e}, W={wil.statistic}')
print(f'cpsat wall {summary["cpsat_Cmax"]["wall_s_mean"]:.2f}s mean, proven optimal {summary["cpsat_Cmax"]["proven_optimal"]}/{N_INS}')
print('saved q2_family_results.csv / q2_family_summary.json')
