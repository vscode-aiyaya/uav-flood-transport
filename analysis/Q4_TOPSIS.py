# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。AI工具名称：DeepSeek；版本/型号：V4.1-Flash；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；版本发布日期：2026年9月10日。

"""Q4 TOPSIS脆弱度（辅助评价）—— 熵权TOPSIS，输出表B.9数据"""
import json
import numpy as np
import pandas as pd
from collections import defaultdict

import os
BASE = os.path.dirname(os.path.abspath(__file__))
sa_data = pd.read_excel(BASE + r'\数据\无人机应急物资运输基础数据\调度中心与服务区.xlsx', header=None)
sa_info = {}
for i in range(6, 21):
    code = str(sa_data.iloc[i, 0]).strip()
    if code.startswith('S'):
        sa_info[code] = dict(lon=float(sa_data.iloc[i, 2]), lat=float(sa_data.iloc[i, 3]),
                             elev=float(sa_data.iloc[i, 4]), pop=float(sa_data.iloc[i, 5]))
goods = pd.read_excel(BASE + r'\数据\无人机应急物资运输基础数据\物资需求与配送时限.xlsx', header=None)
sa_demand = defaultdict(lambda: dict(boxes=0, mass=0, vol=0, first=0))
for i in range(1, len(goods)):
    code = str(goods.iloc[i, 0]).strip()
    if not code.startswith('S'):
        continue
    d = sa_demand[code]
    d['boxes'] += int(goods.iloc[i, 2]); d['mass'] += float(goods.iloc[i, 4]) * int(goods.iloc[i, 2])
    d['vol'] += float(goods.iloc[i, 5]) * int(goods.iloc[i, 2]); d['first'] += int(goods.iloc[i, 3])

O01 = dict(lon=109.2308517, lat=23.0085095)
def hav(lon1, lat1, lon2, lat2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1); dl = np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

ALL_SA = sorted(sa_info.keys())
# 指标矩阵：质量kg, 体积m3, 距离m, 海拔m（需求规模×到达难度，首批箱数方差过小熵权趋0、人口为需求规模非脆弱度，均不纳入）
M = []
for s in ALL_SA:
    M.append([sa_demand[s]['mass'], sa_demand[s]['vol'],
              hav(O01['lon'], O01['lat'], sa_info[s]['lon'], sa_info[s]['lat']),
              sa_info[s]['elev']])
M = np.array(M, dtype=float)
n, m = M.shape

# 正向化（全部为效益型：越大越脆弱）
# 熵权法
def entropy_weight(X):
    # X 已正向化；归一化每列
    P = X / np.maximum(X.sum(axis=0), 1e-12)
    k = 1.0 / np.log(n)
    e = -k * np.sum(P * np.log(np.maximum(P, 1e-12)), axis=0)
    w = (1 - e) / np.maximum(np.sum(1 - e), 1e-12)
    return w

# 归一化（向量归一化）
norm = M / np.sqrt((M ** 2).sum(axis=0))
w = entropy_weight(M)
# 加权决策矩阵
V = norm * w
ideal = V.max(axis=0)
nadir = V.min(axis=0)
d_pos = np.sqrt(((V - ideal) ** 2).sum(axis=1))
d_neg = np.sqrt(((V - nadir) ** 2).sum(axis=1))
score = d_neg / np.maximum(d_pos + d_neg, 1e-12)

# 脆弱度排名
order = np.argsort(-score)
rank = {ALL_SA[i]: r + 1 for r, i in enumerate(order)}

rows = []
for s in ALL_SA:
    rows.append(dict(sa=s, mass=sa_demand[s]['mass'], vol=round(sa_demand[s]['vol'], 4),
                     dist=round(hav(O01['lon'], O01['lat'], sa_info[s]['lon'], sa_info[s]['lat']), 1),
                     elev=sa_info[s]['elev'], first=sa_demand[s]['first'],
                     norm=round(float(norm[ALL_SA.index(s), 0]), 4),
                     closeness=round(float(score[ALL_SA.index(s)]), 4),
                     rank=rank[s]))

print('=== 服务区脆弱度（熵权TOPSIS贴近度）===')
print('权重:', dict(zip(['质量','体积','距离','海拔'], np.round(w, 4))))
for r in sorted(rows, key=lambda x: x['rank']):
    print('  %s: 质量%.0fkg 距离%.0fm 贴近度%.4f 排名%d' % (r['sa'], r['mass'], r['dist'], r['closeness'], r['rank']))

with open(BASE + r'\Q4_TOPSIS.json', 'w', encoding='utf-8') as f:
    json.dump(dict(weights=dict(zip(['质量','体积','距离','海拔','首批','人口'], np.round(w, 4).tolist())), rows=rows),
              f, ensure_ascii=False, indent=1, default=str)
print('\n已保存 Q4_TOPSIS.json')
