# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成。
"""
Q3 鲁棒性核查（补充材料，不修改冻结计划与论文数字）

核查两项：
  A 时序抖动容限
    A1 需求窗整体提前：受“首段到位/段间到位”余量约束，给出各中继架次
       可提前的最大秒数及全网最小值。
    A2 需求窗整体推后：受返航 SOC>=0.20 约束，按悬停功率 1.10 kW、电池 3.2 kWh
       折算各架次可额外悬停的秒数及全网最小值。
  B 能源组件数下界
    4 个架次占用区间 [准备开始s, 充电完成s] 固定，枚举架次到组件的分配，
    给出仍可无冲突排程的最少组件数（组件只影响占用，不影响充电时长）。

输出：outputs_B/Q3_鲁棒性.json
"""
import os, json, itertools
import pandas as pd

_REPO_DEFAULT = r'c:\Users\Administrator\AppData\Local\Temp\论文投稿改造\repo\uav-flood-transport\analysis'
BASE = os.environ.get('Q3_REPO', _REPO_DEFAULT)
OUT = os.environ.get('Q3_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs_B'))
SOC_MIN, E_BAT, P_HOV = 0.20, 3.2, 1.10

tt = pd.read_csv(os.path.join(OUT, 'Q3_固定方案_悬停点时窗.csv'), encoding='utf-8-sig')
sched = pd.read_csv(BASE + r'\Q3_两架方案_中继排程.csv', encoding='utf-8-sig')
ver = json.load(open(os.path.join(OUT, 'Q3_只读验证.json'), encoding='utf-8'))

# ---------- A1 可提前量 ----------
slack = {k: float(v) for k, v in ver['首段到位余量s'].items()}
for _, r in tt[tt['到位余量s'].notna()].iterrows():
    k = '%s-s%d' % (r['任务ID'], int(r['时段']))
    slack[k] = float(r['到位余量s'])
min_adv = min(slack.values())
min_adv_key = min(slack, key=slack.get)

# ---------- A2 可推后量（SOC 约束） ----------
soc_df = pd.read_csv(os.path.join(OUT, 'Q3_核验方案_能耗SOC对照_eta0.72.csv'), encoding='utf-8-sig')
extra = {}
for _, r in soc_df.iterrows():
    extra[r['任务ID']] = (float(r['返航SOC_核验方案'])-SOC_MIN)*E_BAT*3600.0/P_HOV
min_delay = min(extra.values())
min_delay_key = min(extra, key=extra.get)

# ---------- B 组件数 ----------
iv = [(r['任务ID'], float(r['准备开始s']), float(r['充电完成s'])) for _, r in sched.iterrows()]
def feasible(k):
    for asg in itertools.product(range(k), repeat=len(iv)):
        occ = {c: [] for c in range(k)}
        ok = True
        for (mid, s, e), c in zip(iv, asg):
            for s2, e2 in occ[c]:
                if s < e2 and s2 < e:
                    ok = False
                    break
            if not ok:
                break
            occ[c].append((s, e))
        if ok:
            return asg
    return None
kmin, asg = None, None
for k in range(1, len(iv)+1):
    a = feasible(k)
    if a is not None:
        kmin, asg = k, a
        break

res = dict(
    可提前s=dict(sorted(slack.items(), key=lambda z: z[1])),
    最小可提前s=round(min_adv, 1), 最小可提前对象=min_adv_key,
    可推后s={k: round(v, 0) for k, v in extra.items()},
    最小可推后s=round(min_delay, 0), 最小可推后对象=min_delay_key,
    最少能源组件数=kmin,
    组件分配示例={mid: 'C%d' % (c+1) for (mid, _, _), c in zip(iv, asg)} if asg else None,
)
json.dump(res, open(os.path.join(OUT, 'Q3_鲁棒性.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print(json.dumps(res, ensure_ascii=False, indent=2))
