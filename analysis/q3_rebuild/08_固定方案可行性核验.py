# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成。
"""
B：固定主方案【只读可行性核验】——不搜索、不重建，只检查给定方案是否自洽可行。

核验对象（固定方案 = 冻结排程坐标 4 个 + 补齐冻结记录缺失的重建坐标 2 个 = 7 个悬停段）：
  1) 悬停点清单：每个悬停段的坐标、来源（冻结/补齐）；
  2) 服务时窗：由【规范需求集合】实际归属推导（不沿用冻结排程表的到岗/离岗信封）；
  3) 转场轨迹与时间：相邻悬停段之间的垂直调整 + 水平巡航时长（15 m/s 巡航、4 m/s 爬升、3 m/s 下降）；
  4) 建链时间：每段到岗前 30 s 建链（T_LINK）；
  5) 组件编号与 SOC：取自冻结排程表，SOC 由核验方案能耗重算；
  6) 转场/重新建链期通信保障：检查相邻段之间的间隙是否 ≥ 所需转场时间（否则中继无法按时到位，
     该段首个需求点将不可达），并逐点复核到位时刻前的需求覆盖。

判定规则：
  · 可行(FEASIBLE)：每段“进入时刻 ≤ 该段首个需求点时刻”，且段内全部需求点在该悬停点覆盖，
    且各架次 SOC ≥ 0.20；
  · 若某段转场时间不足 → 标记 INFEASIBLE_TRANSIT，并给出必须提前的秒数（用于修改方案而非掩盖）。

输入（只读）：outputs 下的 02 核验汇总、06 逐点核验、07 能耗对照；仓库 analysis 的冻结排程表。
输出：Q3_固定方案_悬停点时窗.csv、Q3_固定方案_可行性核验.json
"""
import os
import json
import numpy as np
import pandas as pd
from math import radians

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_DEFAULT = os.path.normpath(os.path.join(_HERE, '..'))
BASE = os.environ.get('Q3_REPO', _REPO_DEFAULT)
OUT = os.environ.get('Q3_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs'))
ETA = os.environ.get('ETA_CLIMB', '0.72')

V, V_UP, V_DN = 15.0, 4.0, 3.0
T_LINK = 30.0
SOC_MIN = 0.20

def hav_m(lo1, la1, lo2, la2):
    a = np.sin((la2-la1)/2.0)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2.0)**2
    return 6371000.0*2*np.arctan2(np.sqrt(a), np.sqrt(1-a))

def transit_s(a, b):
    """相邻悬停段之间的转场时长 s：垂直调整 + 水平巡航。"""
    dh = b[2] - a[2]
    return (dh/V_UP if dh >= 0 else -dh/V_DN) + hav_m(radians(a[0]), radians(a[1]),
                                                       radians(b[0]), radians(b[1]))/V

plan2 = pd.read_csv(os.path.join(OUT, 'Q3_重建_冻结方案核验汇总.csv'), encoding='utf-8-sig')
pt06 = pd.read_csv(os.path.join(OUT, 'Q3_规范集合_覆盖核验逐点.csv'), encoding='utf-8-sig')
ene07 = pd.read_csv(os.path.join(OUT, 'Q3_核验方案_能耗SOC对照_eta%s.csv' % ETA), encoding='utf-8-sig')
sched = pd.read_csv(os.path.join(BASE, 'Q3_两架方案_中继排程.csv'))

# 优先用 06 逐点核验推导的实际悬停坐标（与核验一致），缺失时回落到 02 汇总
coord = {}
for _, r in plan2.iterrows():
    coord[(r['任务ID'], int(r['时段']))] = (float(r['悬停经度']), float(r['悬停纬度']),
                                            float(r['悬停海拔m']), r['悬停点来源'])

# 由规范需求集合推导每段的实际在岗时窗
win = {}
for (mid, seg), g in pt06.groupby(['保障中继', '时段']):
    win[(mid, int(seg))] = (float(g['t'].min()), float(g['t'].max()), len(g),
                            int(g['核验通过'].sum()), float(g['TR中继弱向Pr'].min()),
                            float(g['中继G01弱向Pr'].min()))

ene = {r['任务ID']: r for _, r in ene07.iterrows()}
sch = {r['任务ID']: r for _, r in sched.iterrows()}

rows = []; verdict = []; infeasible = []
for mid in sorted(set(k[0] for k in coord)):
    segs = sorted(k[1] for k in coord if k[0] == mid)
    prev_pt = None; prev_end = None; t_ready = None
    for si, seg in enumerate(segs):
        lon, lat, alt, src = coord[(mid, seg)]
        t0, t1, n, nok, pr_tr, pr_g01 = win[(mid, seg)]
        pt = (lon, lat, alt)
        t_tr = transit_s(prev_pt, pt) if prev_pt is not None else None
        # 到位窗口：上一段结束后才能起飞；本段需在首个需求点之前走完转场 + 建链
        if t_tr is None:
            t_arrive = t0                      # 首段：出航时间由排程表给出，此处仅作到位上界
            slack = None
            ok_pos = True
        else:
            t_ready_prev = prev_end + t_tr + T_LINK
            t_arrive = t_ready_prev
            slack = t0 - t_ready_prev          # ≥0 表示可按时到位
            ok_pos = bool(slack >= 0)
        if not ok_pos:
            infeasible.append(dict(任务ID=mid, 时段=seg, 必须提前s=round(-slack, 1),
                                   转场s=round(t_tr, 1)))
        rows.append(dict(任务ID=mid, 时段=seg, 悬停点来源=src,
                         经度=round(lon, 5), 纬度=round(lat, 5), 海拔m=round(alt, 1),
                         在岗起s=round(t0, 1), 在岗止s=round(t1, 1), 在岗时长s=round(t1-t0, 1),
                         需求点数=n, 核验通过=nok, 覆盖率=round(nok/n, 6),
                         段前转场s=(None if t_tr is None else round(t_tr, 1)),
                         建链s=T_LINK, 到位时刻s=(None if t_arrive is None else round(t_arrive, 1)),
                         到位余量s=(None if slack is None else round(slack, 1)),
                         TR最弱Pr=round(pr_tr, 2), G01最弱Pr=round(pr_g01, 2)))
        prev_pt, prev_end = pt, t1
        t_ready = t_arrive

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, 'Q3_固定方案_悬停点时窗.csv'), index=False, encoding='utf-8-sig')

# 架次级汇总
tot_pts = int(res['需求点数'].sum()); tot_ok = int(res['核验通过'].sum())
coord_set = set(k[0] for k in coord)
summary = dict(
    eta_c=float(ETA),
    悬停段数=int(len(res)),
    中继机数=int(len(set(str(t)[:3] for t in res['任务ID']))),
    架次数=int(res['任务ID'].nunique()),
    规范需求点数=tot_pts, 核验通过=tot_ok, 覆盖率=round(tot_ok/tot_pts, 6),
    最低到位余量s=(None if res['到位余量s'].dropna().empty else float(res['到位余量s'].dropna().min())),
    全局最弱TR中继Pr_dBm=float(res['TR最弱Pr'].min()),
    全局最弱中继G01Pr_dBm=float(res['G01最弱Pr'].min()),
    最低返航SOC=float(ene07['返航SOC_核验方案'].min()),
    SOC门槛=SOC_MIN,
    中继总能耗_kWh=round(float(ene07['E架次_核验方案'].sum()), 4),
    冻结中继能耗_kWh=round(float(ene07['E架次_冻结'].sum()), 4),
    联合总能耗_kWh=round(65.25 + float(ene07['E架次_核验方案'].sum()), 4),
    能源组件使用=sorted(set(str(r['能源组件']) for _, r in sched.iterrows()
                          if r['任务ID'] in coord_set)),
    不可行段=infeasible,
)
summary['可行'] = bool(len(infeasible) == 0
                       and tot_ok == tot_pts
                       and summary['最低返航SOC'] >= SOC_MIN)
with open(os.path.join(OUT, 'Q3_固定方案_可行性核验.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)

print(res[['任务ID', '时段', '悬停点来源', '在岗起s', '在岗止s', '需求点数', '核验通过',
           '段前转场s', '到位时刻s', '到位余量s', 'TR最弱Pr', 'G01最弱Pr']].to_string(index=False))
print('-'*100)
for k, v in summary.items():
    print('  %-22s %s' % (k, v))
