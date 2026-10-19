# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成。
"""
A1（修订版）：按【已核验的固定主方案】重算中继能耗与 SOC —— 使用先验参数 eta_c=0.72。

与脚本03的区别（本次投稿修订新增）：
  · 脚本03 用 reconstruct() 在候选网格上**搜索**悬停点链条，其坐标与脚本02/06
    核验所用的坐标并不相同（"算能耗用的点"≠"核验过的点"），且 eta_c=0.50 系由
    冻结单悬停架次转场能耗**逆向标定**而来，属"含参数标定的重建检查"，不是独立验证；
  · 本脚本的悬停链条**只读**取自脚本02输出的《核验汇总》（4 个冻结坐标 + 2 个重建
    补齐坐标 = 7 个悬停段），与脚本06的覆盖核验使用**同一套坐标**；
  · eta_c 取论文正文式5.2 的先验模型值 0.72（非拟合值），可经环境变量 ETA_CLIMB 覆盖。

能耗模型（论文式5.2，与脚本03一致）：
  出航 = 爬升(O01→首悬停高) + 巡航(O01→首悬停点)
  段间转场 = Σ_{相邻悬停段}(垂直调整时长 + 水平巡航时长)
  返航 = 巡航(末悬停点→O01) + 下降(末悬停高→O01海拔)
  E转场 = P_CR*T飞行/3600 + m*g*ΣΔh_爬升/(eta_c*3.6e6)
  E悬停 = P_HOV*((离岗-到岗)-T段间转场)/3600 ; E架次 = E转场+E悬停
  SOC = 1 - E架次/3.2 ，门槛 SOC>=0.20

输出：Q3_核验方案_能耗SOC对照_eta<p>.csv
"""
import os
import numpy as np
import pandas as pd
from math import radians

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_DEFAULT = os.path.normpath(os.path.join(_HERE, '..'))
BASE = os.environ.get('Q3_REPO', _REPO_DEFAULT)
OUT = os.environ.get('Q3_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs'))

# ---- 能耗参数（题目/论文式5.2） ----
G_M, MASS = 9.81, 23.5
ETA_CLIMB = float(os.environ.get('ETA_CLIMB', '0.72'))
P_CR, P_HOV, E_BAT = 1.15, 1.10, 3.2
V, V_UP, V_DN = 15.0, 4.0, 3.0
O_ALT = 127.7

def hav_m(lo1, la1, lo2, la2):
    a = np.sin((la2-la1)/2.0)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2.0)**2
    return 6371000.0*2*np.arctan2(np.sqrt(a), np.sqrt(1-a))

_no = pd.read_csv(os.path.join(BASE, '预处理输出', 'Q1_节点_预处理.csv'))
_o = _no[_no['编号'] == 'O01'].iloc[0]
O01 = (float(_o['经度']), float(_o['纬度']))

sched = pd.read_csv(os.path.join(BASE, 'Q3_两架方案_中继排程.csv'))
plan = pd.read_csv(os.path.join(OUT, 'Q3_重建_冻结方案核验汇总.csv'), encoding='utf-8-sig')

print('eta_c = %.2f（0.72=论文先验值；0.50=冻结值逆向标定口径）' % ETA_CLIMB)
print('悬停链条来源：脚本02《核验汇总》（只读，与脚本06核验坐标一致）')
chain = {}
for mid, g in plan.groupby('任务ID'):
    g = g.sort_values('时段')
    chain[mid] = [(float(r['悬停经度']), float(r['悬停纬度']), float(r['悬停海拔m']))
                  for _, r in g.iterrows()]
    print('  %s: %s' % (mid, [(round(c[0], 5), round(c[1], 5), round(c[2], 1)) for c in chain[mid]]))

rows = []
for _, r in sched.iterrows():
    mid = r['任务ID']; pts = chain[mid]
    first, last = pts[0], pts[-1]
    t_out = max(first[2]-O_ALT, 0.0)/V_UP + hav_m(radians(O01[0]), radians(O01[1]),
                                                  radians(first[0]), radians(first[1]))/V
    t_trans = 0.0; dh_up = max(first[2]-O_ALT, 0.0)
    for a, b in zip(pts, pts[1:]):
        dh = b[2]-a[2]
        t_trans += (dh/V_UP if dh >= 0 else -dh/V_DN) + hav_m(radians(a[0]), radians(a[1]),
                                                              radians(b[0]), radians(b[1]))/V
        if dh > 0:
            dh_up += dh
    t_back = hav_m(radians(last[0]), radians(last[1]), radians(O01[0]), radians(O01[1]))/V \
             + max(last[2]-O_ALT, 0.0)/V_DN
    t_fly = t_out + t_trans + t_back
    e_fly = P_CR*t_fly/3600.0 + G_M*MASS*dh_up/(ETA_CLIMB*3.6e6)
    srv = float(r['离岗时刻s']) - float(r['到岗时刻s'])
    e_hov = P_HOV*(srv - t_trans)/3600.0
    e_tot = e_fly + e_hov
    soc = 1.0 - e_tot/E_BAT
    rows.append(dict(任务ID=mid, 悬停段数=len(pts),
                     T出航s=round(t_out, 1), T段间转场s=round(t_trans, 1), T返航s=round(t_back, 1),
                     E转场_核验方案=round(e_fly, 4), E转场_冻结=float(r['转场能耗kWh']),
                     E悬停_核验方案=round(e_hov, 4), E悬停_冻结=float(r['悬停能耗kWh']),
                     E架次_核验方案=round(e_tot, 4), E架次_冻结=float(r['架次能耗kWh']),
                     能源组件=r['能源组件'], 返航SOC_核验方案=round(soc, 4),
                     返航SOC_冻结=float(r['返航SOC']),
                     架次能耗差pct=round(100*(e_tot-float(r['架次能耗kWh']))/float(r['架次能耗kWh']), 2),
                     SOC达标=('是' if soc >= 0.20 else '否')))
res = pd.DataFrame(rows)
tag = os.environ.get('Q3_TAG', 'eta%.2f' % ETA_CLIMB)
res.to_csv(os.path.join(OUT, 'Q3_核验方案_能耗SOC对照_%s.csv' % tag), index=False, encoding='utf-8-sig')

print(res[['任务ID', '悬停段数', 'E转场_核验方案', 'E转场_冻结', 'E悬停_核验方案', 'E悬停_冻结',
           'E架次_核验方案', 'E架次_冻结', '架次能耗差pct', '返航SOC_核验方案', '返航SOC_冻结',
           'SOC达标', '能源组件']].to_string(index=False))
print('-'*78)
tot_new, tot_frz = res['E架次_核验方案'].sum(), res['E架次_冻结'].sum()
print('中继总能耗：核验方案 %.4f kWh，冻结 %.4f kWh（论文正文 5.34），差 %+.2f%%'
      % (tot_new, tot_frz, 100*(tot_new-tot_frz)/tot_frz))
print('最低返航 SOC：核验方案 %.4f，冻结 %.4f（门槛 0.20）'
      % (res['返航SOC_核验方案'].min(), res['返航SOC_冻结'].min()))
print('联合总能耗（冻结运输 65.25 + 核验中继 %.4f）= %.4f kWh' % (tot_new, 65.25+tot_new))
