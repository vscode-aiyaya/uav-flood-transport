# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。AI工具名称：DeepSeek；版本/型号：V4.1-Flash；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；版本发布日期：2026年9月10日。

"""
Q2 四指标非支配筛选（Q2_四指标非支配.py）
在全部可行方案（硬超时=0）中，以架次数N、总能耗E、最晚返航Cmax、其他物资软延误Lsoft
四类指标做非支配筛选，输出非支配方案集。支撑论文4.3"四指标非支配筛选"表述。
用法: python Q2_四指标非支配.py
"""
import importlib.util, sys, os
import pandas as pd

spec = importlib.util.spec_from_file_location('q2', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Q2_精确调度.py'))
mod = importlib.util.module_from_spec(spec)
sys.modules['q2'] = mod
spec.loader.exec_module(mod)

feasible = [r for r in mod.results if r['hard_late'] == 0]
print('可行方案总数:', len(feasible))

def dom(a, b):
    """a 支配 b：四指标均不差且至少一项严格更优"""
    ka = (a['N'], a['E'], a['Cmax'], a['soft_min'])
    kb = (b['N'], b['E'], b['Cmax'], b['soft_min'])
    return all(x <= y for x, y in zip(ka, kb)) and any(x < y for x, y in zip(ka, kb))

non_dom = []
for r in feasible:
    if not any(dom(o, r) for o in feasible):
        non_dom.append(r)

print('四指标非支配方案数:', len(non_dom))
rows = []
for r in sorted(non_dom, key=lambda x: (x['N'], x['E'])):
    rows.append(dict(架次数=r['N'], 总能耗kWh=round(r['E'], 4),
                     Cmax_h=round(r['Cmax'] / 3600, 4), 软延误min=round(r['soft_min'], 2)))
    print('N=%d E=%.4f kWh Cmax=%.3f h Lsoft=%.2f min' % (r['N'], r['E'], r['Cmax'] / 3600, r['soft_min']))
pd.DataFrame(rows).to_csv(os.path.join(OUT, r'Q2_四指标非支配方案.csv'),
                          index=False, encoding='utf-8-sig')
print('已输出 Q2_四指标非支配方案.csv')
