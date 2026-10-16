# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。AI工具名称：DeepSeek；版本/型号：V4.1-Flash；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；版本发布日期：2026年9月10日。

"""Q2 基线重建 + 新前沿主方案全套输出（19架次/20架次，时间标签与编号统一）"""
import importlib.util, sys
import pandas as pd
import os
spec = importlib.util.spec_from_file_location('q2', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Q2_精确调度.py'))
mod = importlib.util.module_from_spec(spec)
sys.modules['q2'] = mod
spec.loader.exec_module(mod)
import os
OUT = os.path.dirname(os.path.abspath(__file__))
cargo = mod.cargo
hard_deadline = mod.hard_deadline
pps = mod.pps

# 程序内部编号 -> 题目原始无人机编号
UAV_MAP = {f'{mt}{i}': f'U0{m_i}' for mt, start in [('A', 1), ('B', 5), ('C', 7)] for i, m_i in zip(range(1, 5), range(start, start + 4))}
UAV_MAP = {k: v for k, v in UAV_MAP.items() if k[0] in 'ABC' and int(k[1:]) <= {'A': 4, 'B': 2, 'C': 2}[k[0]]}

# ---------- 单区18架次基线重建（Q2口径，作不可行对照） ----------
base_tasks = []
for _, r in mod.q1.iterrows():
    zb = {r['服务区']: [str(x).strip() for x in str(r['货箱编号']).split(';')]}
    t = mod.mk_task([r['服务区']], zb, [r['机型']])
    assert t is not None, f'基线任务失败: {r["服务区"]} {r["机型"]}'
    base_tasks.append(t)
br = mod.plan_value(base_tasks)
print('=== 单区18架次基线（Q2口径重建） ===')
print(f'N={br["N"]} E={br["E"]:.4f}kWh Cmax={br["Cmax"]/3600:.3f}h 硬超时={br["hard_late"]} 软延误={br["soft_min"]:.1f}min')
for b, dl, arr in br['hard_detail']:
    print(f'  超时: {b} 截止{dl}s 实际{arr}s')
pd.DataFrame([dict(方案='单区18基线', N=br['N'], E=round(br['E'], 4), Cmax_h=round(br['Cmax']/3600, 4),
                   hard_late=br['hard_late'], soft_min=round(br['soft_min'], 2),
                   hard_detail='; '.join(f'{b}@{dl}' for b, dl, _ in br['hard_detail']))]).to_csv(
    os.path.join(OUT, 'Q2_基线_单区18.csv'), index=False, encoding='utf-8-sig')

# ---------- 主方案全套输出（19/20架次新前沿） ----------
def dump_plan(r, tag):
    sched = r['sched']
    n_total_boxes = 80
    trip_rows = []
    for idx, s in enumerate(sched, start=1):
        mt = s['mt']
        pp = pps[mt]
        w = sum(cargo[cargo['货箱序号'] == b]['单箱质量(kg)'].iloc[0] for b in s['boxes'])
        n = len(s['boxes'])
        takeoff = s['dep'] + pp['t_prep'] + pp['t_load'] * n
        trip_rows.append(dict(架次ID=f'T{idx:02d}', 无人机=UAV_MAP[s['uav']], 机型=mt,
                             路径='→'.join(s['seq']), 货箱数=n, 载重kg=round(w, 1),
                             准备开始s=round(s['dep'], 1), 起飞s=round(takeoff, 1),
                             交付完成={z: round(t, 1) for z, t in s['arr'].items()},
                             返航s=round(s['ret'], 1), 能耗kWh=round(s['E'], 4),
                             返航SOC=round(1.0 - s['E'] / pp['E'], 4), 电池充满s=round(s['bat_ready'], 1)))
    pd.DataFrame(trip_rows).to_csv(os.path.join(OUT, f'Q2_{tag}_架次表.csv'), index=False, encoding='utf-8-sig')

    box_rows = []
    for idx, s in enumerate(sched, start=1):
        for b in s['boxes']:
            z = next(z for z in s['zones'] if b in cargo[cargo['服务区编号'] == z]['货箱序号'].tolist())
            row = cargo[cargo['货箱序号'] == b].iloc[0]
            box_rows.append(dict(货箱序号=b, 服务区=z, 物资类型=row['物资类型'],
                                 质量kg=float(row['单箱质量(kg)']), 体积m3=float(row['单箱体积(m³)']),
                                 是否首批='是' if bool(row['是否首批']) else '否',
                                 硬截止s=hard_deadline.get(b, ''),
                                 期望送达s=float(row['期望送达时间(s)']),
                                 交付完成s=round(s['arr'][z], 1),
                                 是否超时='是' if b in hard_deadline and s['arr'][z] > hard_deadline[b] + 1e-9 else '否',
                                 架次ID=f'T{idx:02d}', 无人机=UAV_MAP[s['uav']], 机型=s['mt']))
    pd.DataFrame(box_rows).to_csv(os.path.join(OUT, f'Q2_{tag}_逐箱交付.csv'), index=False, encoding='utf-8-sig')

    uav_rows = []
    for idx, s in enumerate(sched, start=1):
        pp = pps[s['mt']]
        n = len(s['boxes'])
        takeoff = s['dep'] + pp['t_prep'] + pp['t_load'] * n
        uav_rows.append(dict(无人机=UAV_MAP[s['uav']], 机型=s['mt'], 服务区='/'.join(s['zones']),
                             架次ID=f'T{idx:02d}',
                             准备开始s=round(s['dep'], 1), 起飞s=round(takeoff, 1),
                             返航s=round(s['ret'], 1),
                             电池=s['bat'][0] + '#' + str(s['bat'][1]),
                             电池充满s=round(s['bat_ready'], 1)))
    pd.DataFrame(uav_rows).to_csv(os.path.join(OUT, f'Q2_{tag}_无人机时刻表.csv'), index=False, encoding='utf-8-sig')
    print(f'{tag}: N={r["N"]} E={r["E"]:.4f}kWh Cmax={r["Cmax"]/3600:.3f}h 硬超时={r["hard_late"]} 软延误={r["soft_min"]:.1f}min')

feasible = [r for r in mod.results if r['hard_late'] == 0]
# 取新前沿两个非支配点：N=19 与 N=20（能耗最低者）
byN = {}
for r in feasible:
    if r['N'] not in byN or r['E'] < byN[r['N']]['E']:
        byN[r['N']] = r
for N in sorted(byN):
    print(f'可行方案 N={N}: E={byN[N]["E"]:.4f} Cmax={byN[N]["Cmax"]/3600:.3f}h 软延误={byN[N]["soft_min"]:.1f}min')
if 19 in byN:
    dump_plan(byN[19], '主方案19架次')
if 20 in byN:
    dump_plan(byN[20], '主方案20架次')
if 21 in byN:
    dump_plan(byN[21], '主方案21架次')
print('主方案全套CSV已导出')
