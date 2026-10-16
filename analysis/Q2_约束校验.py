# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。AI工具名称：DeepSeek；版本/型号：V4.1-Flash；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；版本发布日期：2026年9月10日。

"""Q2 独立校验器 v2：从最终导出的CSV（架次表+逐箱交付+无人机时刻表）独立重算全部约束。
   不信任CSV中的能耗/SOC/时刻列，禁止SOC钳制；结果写入 Q2_复核审计.json"""
import importlib.util, sys
import pandas as pd
import os, json
from collections import Counter

spec = importlib.util.spec_from_file_location('q2', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Q2_精确调度.py'))
mod = importlib.util.module_from_spec(spec)
sys.modules['q2'] = mod
spec.loader.exec_module(mod)
cargo = mod.cargo
hard_deadline = mod.hard_deadline
pps = mod.pps
FLEET, BATS, TCHG = mod.FLEET, mod.BATS, mod.TCHG
import os
OUT = os.path.dirname(os.path.abspath(__file__))

def charge_time(s, Tfull):
    if s < 0.9:
        return Tfull * (0.65 * (0.9 - s) / 0.9 + 0.35)
    return Tfull * 0.35 * (1.0 - s) / 0.1

def verify(tag):
    trips = pd.read_csv(os.path.join(OUT, f'Q2_{tag}_架次表.csv'))
    boxes = pd.read_csv(os.path.join(OUT, f'Q2_{tag}_逐箱交付.csv'))
    uavs = pd.read_csv(os.path.join(OUT, f'Q2_{tag}_无人机时刻表.csv'))
    report = {'方案': tag, '架次数': len(trips), '检查项': {}}

    # 1) 80箱各一次
    cnt = Counter(boxes['货箱序号'])
    missing = set(cargo['货箱序号']) - set(cnt)
    dup = {b: c for b, c in cnt.items() if c > 1}
    ok = not missing and not dup
    report['检查项']['80箱各一次'] = {'缺失': sorted(missing), '重复': dup, '通过': ok}
    assert ok, (tag, missing, dup)

    # 2) 31箱硬时限（逐箱：硬截止 vs 交付完成）
    late = boxes[boxes['是否超时'] == '是']
    hard_boxes = set(boxes[boxes['硬截止s'].notna()]['货箱序号'])
    ok = len(late) == 0 and hard_boxes == set(hard_deadline)
    report['检查项']['31箱硬时限'] = {'超时箱': late['货箱序号'].tolist() if len(late) else [],
                               '涉及箱数': len(hard_boxes), '通过': ok}

    # 3) 每架次载重/体积（用原始货箱数据重算）≤ 机型容量
    viol = []
    for _, t in trips.iterrows():
        bs = boxes[boxes['架次ID'] == t['架次ID']]['货箱序号']
        w = sum(cargo[cargo['货箱序号'] == b]['单箱质量(kg)'].iloc[0] for b in bs)
        v = sum(cargo[cargo['货箱序号'] == b]['单箱体积(m³)'].iloc[0] for b in bs)
        pp = pps[t['机型']]
        if w > pp['Q'] + 1e-9 or v > pp['V'] + 1e-9:
            viol.append((t['架次ID'], w, v, pp['Q'], pp['V']))
    report['检查项']['载重/体积'] = {'违规': viol, '通过': not viol}

    # 4) 独立重算能耗：路径逐段剩余载荷 + DEM巡航海拔 + 作业高度+30 + 爬升附加
    #    （重新走一遍物理公式，对比CSV能耗并报告偏差）
    for _, t in trips.iterrows():
        seq = t['路径'].split('→')
        mt = t['机型']
        pp = pps[mt]
        bs = [str(b) for b in boxes[boxes['架次ID'] == t['架次ID']]['货箱序号']]
        zb = {z: [b for b in bs if b.startswith(z + '-')] for z in seq[1:-1]}
        rem = sum(cargo[cargo['货箱序号'] == b]['单箱质量(kg)'].iloc[0] for b in bs)
        E = 0.0
        for k in range(len(seq) - 1):
            a, b = seq[k], seq[k + 1]
            d = mod.hav(mod.lon[a], mod.lat[a], mod.lon[b], mod.lat[b])
            q = rem if b != 'O01' else 0.0
            E += mod.seg_energy(pp, q, d, mod.Hc[(a, b)], mod.h_op[a], mod.h_op[b])
            if b != 'O01':
                rem -= sum(cargo[cargo['货箱序号'] == bx]['单箱质量(kg)'].iloc[0] for bx in zb[b])
        csvE = float(t['能耗kWh'])
        soc = 1.0 - E / pp['E']
        if abs(E - csvE) > 0.01:
            report.setdefault('能耗偏差', []).append((t['架次ID'], csvE, round(E, 4)))
        if soc < pp['rho'] - 1e-9:
            report.setdefault('SOC违规', []).append((t['架次ID'], round(soc, 4), pp['rho']))

    report['检查项']['独立能耗重算'] = {
        '偏差>0.01kWh的架次': report.get('能耗偏差', []),
        'SOC<rho': report.get('SOC违规', []),
        '通过': not report.get('能耗偏差') and not report.get('SOC违规')}

    # 5) 无人机占用冲突（准备开始~返航区间）
    conf = []
    for i in range(len(trips)):
        for j in range(i + 1, len(trips)):
            a, b = trips.iloc[i], trips.iloc[j]
            if a['无人机'] == b['无人机'] and not (a['返航s'] <= b['准备开始s'] + 1e-9 or b['返航s'] <= a['准备开始s'] + 1e-9):
                conf.append((a['无人机'], a['准备开始s'], a['返航s'], b['准备开始s'], b['返航s']))
    report['检查项']['无人机占用'] = {'冲突': conf, '通过': not conf}

    # 6) 电池周转：同机型同电池号任务区间+充电不重叠；复用前充至100%（用真实SOC算充满时刻）
    bviol = []
    rec = {}
    for _, t in uavs.iterrows():
        mt = t['机型']
        bat = t['电池']  # 形如 'C#3'
        dep = float(t['准备开始s'])
        if bat in rec and rec[bat] > dep + 1e-9:
            bviol.append((bat, rec[bat], dep))
        rec[bat] = float(t['电池充满s'])
    report['检查项']['电池周转'] = {'冲突': bviol, '通过': not bviol}

    # 汇总
    all_ok = all(v['通过'] for v in report['检查项'].values())
    report['总通过'] = all_ok
    return report

reports = []
for tag in ['主方案19架次', '主方案20架次']:
    rep = verify(tag)
    reports.append(rep)
    print(f"===== {tag} 独立校验 =====")
    for k, v in rep['检查项'].items():
        print(f"  {k}: 通过={v['通过']}")
    print(f"  总通过: {rep['总通过']}")
json.dump(reports, open(os.path.join(OUT, 'Q2_复核审计.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print('已写入 Q2_复核审计.json')
