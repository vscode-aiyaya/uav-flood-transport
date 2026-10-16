# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。AI工具名称：DeepSeek；版本/型号：V4.1-Flash；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；版本发布日期：2026年9月10日。

"""Q4 全套图表生成 v3（0836复核修订基准：新Q3完整时间线 + c_h修复 + 全局中继机2架）"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from collections import defaultdict

for f in ['Microsoft YaHei', 'SimHei']:
    try:
        font_manager.findfont(f)
        plt.rcParams['font.sans-serif'] = [f]
        break
    except Exception:
        pass
plt.rcParams['axes.unicode_minus'] = False

import os
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = BASE + r'\Q4_图'
os.makedirs(OUT, exist_ok=True)

R = json.load(open(BASE + r'\Q4_分区核算结果_v3.json', encoding='utf-8'))
T = json.load(open(BASE + r'\Q4_TOPSIS.json', encoding='utf-8'))
g_need = R['global_need']
K2, K3 = R['K2'], R['K3']

ORDER_K2 = [['S001','S002','S003','S004','S005','S006','S007','S008','S009','S011','S015'],
            ['S010','S012','S013','S014']]
ORDER_K3 = [['S001','S002','S004','S005','S008','S009','S011'],
            ['S003','S006','S007','S015'],
            ['S010','S012','S013','S014']]
def reorder_parts(order, detail):
    d_by_sa = {}
    for g in detail:
        for s in g['sas']:
            d_by_sa[s] = g
    out = []
    for grp in order:
        dd = d_by_sa[grp[0]]
        out.append(dd)
    return out
K2_d = reorder_parts(ORDER_K2, K2['detail'])
K3_d = reorder_parts(ORDER_K3, K3['detail'])

sa_data = pd.read_excel(BASE + r'\数据\无人机应急物资运输基础数据\调度中心与服务区.xlsx', header=None)
sa_info = {}
for i in range(6, 21):
    code = str(sa_data.iloc[i, 0]).strip()
    if code.startswith('S'):
        sa_info[code] = dict(lon=float(sa_data.iloc[i, 2]), lat=float(sa_data.iloc[i, 3]),
                             elev=float(sa_data.iloc[i, 4]), pop=float(sa_data.iloc[i, 5]))
O01 = dict(lon=109.2308517, lat=23.0085095)

def plot_partition(parts, title, fname, colors=None):
    fig, ax = plt.subplots(figsize=(9.5, 6.5), dpi=150)
    pal = colors if colors else ['#e63b2e', '#2e7dd1', '#2ea84f', '#8e5cd9']
    for gi, grp in enumerate(parts):
        xs = [sa_info[s]['lon'] for s in grp]
        ys = [sa_info[s]['lat'] for s in grp]
        ax.scatter(xs, ys, s=95, c=pal[gi], edgecolors='white', linewidths=1.2, zorder=3)
        for s in grp:
            ax.annotate(s, (sa_info[s]['lon'], sa_info[s]['lat']),
                        textcoords='offset points', xytext=(7, 6), fontsize=9)
    ax.scatter([O01['lon']], [O01['lat']], marker='^', s=160, c='#333333',
               edgecolors='white', linewidths=1.2, zorder=4, label='O01 调度中心')
    ax.annotate('O01', (O01['lon'], O01['lat']), textcoords='offset points',
                xytext=(8, 8), fontsize=10, fontweight='bold')
    for gi, grp in enumerate(parts):
        cx = np.mean([sa_info[s]['lon'] for s in grp])
        cy = np.mean([sa_info[s]['lat'] for s in grp])
        ax.annotate('组%d（%d区）' % (gi + 1, len(grp)), (cx, cy),
                    fontsize=11, fontweight='bold', color=pal[gi],
                    ha='center', va='bottom', zorder=2)
    ax.set_xlabel('经度（°）', fontsize=12)
    ax.set_ylabel('纬度（°）', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT + '\\' + fname, bbox_inches='tight')
    plt.close(fig)
    print('saved', fname)

plot_partition(ORDER_K3, '问题四 K=3 分区方案（7+4+4）', 'D22_K3分区散点.png')
plot_partition(ORDER_K2, '问题四 K=2 分区方案（11+4）', 'D23_K2分区散点.png')

# D.24 分区特征对比
fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=150)
metrics = [('箱数', 'boxes'), ('质量kg', 'mass'), ('架次数', 'n_trips'), ('能耗kWh', 'energy')]
for ax, (lab, key) in zip(axes.flat, metrics):
    for K, dl in [('K=2', K2_d), ('K=3', K3_d)]:
        vals = [d[key] for d in dl]
        x = np.arange(len(vals))
        ax.bar(x + (0.2 if K == 'K=2' else 0.0), vals, 0.38,
               label=K, color=('#2e7dd1' if K == 'K=2' else '#e63b2e'))
        for xi, v in zip(x, vals):
            ax.text(xi + (0.2 if K == 'K=2' else 0.0), v, '%.1f' % v,
                    ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(['组%d' % (i + 1) for i in range(len(vals))])
    ax.set_title(lab, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
fig.suptitle('问题四 分区特征对比（工作量维度）', fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT + '\\D24_分区特征对比.png', bbox_inches='tight')
plt.close(fig)
print('saved D24_分区特征对比.png')

# D.25 TOPSIS 脆弱度排名
rows = sorted(T['rows'], key=lambda x: x['rank'])
fig, ax = plt.subplots(figsize=(9.5, 6), dpi=150)
names = [r['sa'] for r in rows]
scores = [r['closeness'] for r in rows]
colors = ['#e63b2e' if s > 0.4 else ('#f0a93b' if s > 0.15 else '#2e9ed1') for s in scores]
ax.barh(names[::-1], scores[::-1], color=colors[::-1], edgecolor='white')
for i, s in enumerate(scores[::-1]):
    ax.text(s + 0.01, i, '%.4f' % s, va='center', fontsize=9)
ax.set_xlabel('TOPSIS贴近度（需求×到达难度综合，越高越脆弱）', fontsize=11)
ax.set_title('问题四 服务区脆弱度排名（熵权TOPSIS辅助评价）', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT + '\\D25_TOPSIS脆弱度排名.png', bbox_inches='tight')
plt.close(fig)
print('saved D25_TOPSIS脆弱度排名.png')

# D.26 资源缺口热力图
RES_KEYS = ['uavA', 'uavB', 'uavC', 'batA', 'batB', 'batC', 'relay', 'relay_comp']
RES_SHORT = ['A机', 'B机', 'C机', 'A电池', 'B电池', 'C电池', '中继机', '中继组件']
gap_mat = np.array([[K2['gap'].get(k, 0) for k in RES_KEYS],
                    [K3['gap'].get(k, 0) for k in RES_KEYS]], dtype=int)
fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=150)
im = ax.imshow(gap_mat, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(RES_SHORT))); ax.set_xticklabels(RES_SHORT, fontsize=10)
ax.set_yticks([0, 1]); ax.set_yticklabels(['K=2', 'K=3'], fontsize=11)
for i in range(2):
    for j in range(8):
        v = gap_mat[i, j]
        ax.text(j, i, str(v), ha='center', va='center',
                color=('white' if v > max(2, gap_mat.max() / 2) else '#333333'),
                fontsize=12, fontweight='bold')
ax.set_title('问题四 分区资源缺口热力图（完整占用口径，相对现有库存）', fontsize=13, fontweight='bold')
fig.colorbar(im, shrink=0.8)
fig.tight_layout()
fig.savefig(OUT + '\\D26_资源缺口热力图.png', bbox_inches='tight')
plt.close(fig)
print('saved D26_资源缺口热力图.png')

# D.27 八类资源需求对比
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
x = np.arange(len(RES_SHORT)); w = 0.26
vals_g = [g_need[k] for k in RES_KEYS]
vals2 = [K2['need'][k] for k in RES_KEYS]
vals3 = [K3['need'][k] for k in RES_KEYS]
ax.bar(x - w, vals_g, w, label='全局基准（不分区）', color='#9aa5b1')
ax.bar(x, vals2, w, label='K=2分区', color='#2e7dd1')
ax.bar(x + w, vals3, w, label='K=3分区', color='#e63b2e')
for xi, (a, b, c) in enumerate(zip(vals_g, vals2, vals3)):
    ax.text(xi - w, a, str(a), ha='center', va='bottom', fontsize=8)
    ax.text(xi, b, str(b), ha='center', va='bottom', fontsize=8)
    ax.text(xi + w, c, str(c), ha='center', va='bottom', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(RES_SHORT, fontsize=10)
ax.set_ylabel('最低配置需求（架/组）', fontsize=11)
ax.set_title('问题四 八类资源最低配置需求对比（完整占用口径峰值并发）', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT + '\\D27_八类资源需求对比.png', bbox_inches='tight')
plt.close(fig)
print('saved D27_八类资源需求对比.png')

# D.28/D.29 组间均衡雷达
trips = pd.read_csv(BASE + r'\Q2_主方案20架次_架次表.csv')
sched = pd.read_csv(BASE + r'\Q2_主方案20架次_无人机时刻表.csv')
trip_meta = {r['架次ID']: dict(battery=r['电池'], bat_full=r['电池充满s']) for _, r in sched.iterrows()}
def sa_of_path(path):
    return [p for p in str(path).split('→') if p not in ('O01', '') and p.strip()]
task_list = []
for _, r in trips.iterrows():
    tid = r['架次ID']; mt = trip_meta[tid]
    task_list.append(dict(tid=tid, mtype=r['机型'], sas=sa_of_path(r['路径']),
        prep=float(r['准备开始s']), ret=float(r['返航s']), energy=float(r['能耗kWh'])))
goods = pd.read_excel(BASE + r'\数据\无人机应急物资运输基础数据\物资需求与配送时限.xlsx', header=None)
sa_demand = defaultdict(lambda: dict(boxes=0, mass=0))
for i in range(1, len(goods)):
    code = str(goods.iloc[i, 0]).strip()
    if not code.startswith('S'):
        continue
    sa_demand[code]['boxes'] += int(goods.iloc[i, 2])
    sa_demand[code]['mass'] += float(goods.iloc[i, 4]) * int(goods.iloc[i, 2])

def radar(parts, K, fname):
    dims = ['货箱数', '物资质量', '架次数', '能耗', '任务时长']
    data = []
    for grp in parts:
        tsel = [t for t in task_list if all(s in grp for s in t['sas'])]
        d = [sum(sa_demand[s]['boxes'] for s in grp), sum(sa_demand[s]['mass'] for s in grp),
             len(tsel), sum(t['energy'] for t in tsel),
             max((t['ret'] for t in tsel), default=0) - min((t['prep'] for t in tsel), default=0)]
        data.append(d)
    data = np.array(data, dtype=float)
    mx = data.max(axis=0) + 1e-9
    norm = data / mx
    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=300, subplot_kw=dict(polar=True))
    pal = ['#e63b2e', '#2e7dd1', '#2ea84f']
    for gi, row in enumerate(norm):
        vals = row.tolist() + row.tolist()[:1]
        ax.plot(angles, vals, 'o-', linewidth=1.8, label='组%d' % (gi + 1), color=pal[gi])
        ax.fill(angles, vals, alpha=0.12, color=pal[gi])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=14)
    ax.set_ylim(0, 1)
    ax.set_title('问题四 %s 组间工作量均衡（归一化）' % K, fontsize=15, fontweight='bold', pad=18)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.08), fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT + '\\' + fname, bbox_inches='tight')
    plt.close(fig)
    print('saved', fname)

radar(ORDER_K2, 'K=2分区', 'D28_均衡雷达K2.png')
radar(ORDER_K3, 'K=3分区', 'D29_均衡雷达K3.png')

# D.30 中继跨组复制（新簇映射：由逐点归属推导）
perpt = pd.read_csv(BASE + r'\Q3_两架方案_逐点通信验证.csv')
relay_sas = defaultdict(set)
for _, p in perpt.iterrows():
    w = p['保障中继']
    if not isinstance(w, str) or not w: continue
    tid_task = w.split('-', 1)[1] if '-' in w else w
    for t in task_list:
        if t['tid'] == p['架次']:
            relay_sas[tid_task].update(t['sas'])
rids = sorted(relay_sas.keys())
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
for ax, (K, parts) in zip(axes, [('K=2', ORDER_K2), ('K=3', ORDER_K3)]):
    sa2g = {}
    for gi, grp in enumerate(parts):
        for s in grp:
            sa2g[s] = gi + 1
    y = np.arange(len(rids))[::-1]
    for i, rid in enumerate(rids):
        grps = sorted(set(sa2g[s] for s in relay_sas[rid]))
        ax.barh(y[i], len(grps), height=0.55, color='#e63b2e' if len(grps) > 1 else '#2e9ed1',
                edgecolor='white')
        ax.text(len(grps) + 0.1, y[i], '组%s（%d组）' % ('/'.join('G%d' % g for g in grps), len(grps)),
                va='center', fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(rids, fontsize=11)
    ax.set_xlim(0, 4)
    ax.set_xlabel('涉及任务组数（>1即需跨组复制）', fontsize=10)
    ax.set_title('%s 中继架次跨组情况' % K, fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
fig.suptitle('问题四 中继保障关联与跨组复制', fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT + '\\D30_中继跨组复制.png', bbox_inches='tight')
plt.close(fig)
print('saved D30_中继跨组复制.png')

# D.31 分区方案对比
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
scen = ['全局基准', 'K=2分区', 'K=3分区']
D = [sum(g_need.values()), K2['D'], K3['D']]
G = [sum(max(0, g_need[k] - R['inventory'][k]) for k in g_need), K2['G'], K3['G']]
x = np.arange(3)
ax.bar(x - 0.16, D, 0.32, label='资源规模D（八类资源需求合计）', color='#2e7dd1')
ax.bar(x + 0.16, G, 0.32, label='库存缺口G', color='#e63b2e')
for xi, d in enumerate(D):
    ax.text(xi - 0.16, d, str(d), ha='center', va='bottom', fontsize=11)
for xi, g in enumerate(G):
    ax.text(xi + 0.16, g, str(g), ha='center', va='bottom', fontsize=11)
ax.set_xticks(x); ax.set_xticklabels(scen, fontsize=12)
ax.set_ylabel('资源单位（架/组）', fontsize=11)
ax.set_title('问题四 分区方案资源规模与缺口对比（完整占用口径）', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT + '\\D31_分区方案对比.png', bbox_inches='tight')
plt.close(fig)
print('saved D31_分区方案对比.png')
# D.24 熵权分配图
w_names = ['物资质量', '物资体积', '距离', '海拔']
w_vals = [T['weights'][k] for k in ['质量', '体积', '距离', '海拔']]
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
bars = ax.bar(w_names, w_vals, color=['#e63b2e', '#f0a93b', '#2e7dd1', '#2ea84f'], edgecolor='white', width=0.55)
for b, v in zip(bars, w_vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.008, '%.4f' % v, ha='center', va='bottom', fontsize=13)
ax.set_ylabel('熵权权重', fontsize=13)
ax.set_title('问题四 熵权TOPSIS服务区评价指标权重', fontsize=15, fontweight='bold')
ax.set_ylim(0, max(w_vals) * 1.18)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT + '\\D24_熵权分配.png', bbox_inches='tight')
plt.close(fig)
print('saved D24_熵权分配.png')

print('\n全部Q4图生成完成(v3)')
