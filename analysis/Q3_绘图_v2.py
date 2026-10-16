# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。AI工具名称：DeepSeek；版本/型号：V4.1-Flash；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；版本发布日期：2026年9月10日。

"""Q3 图重绘 v2（0836修订基准，读 Q3_两架方案/三架方案 冻结CSV）
图5-1 甘特图、图5-2 敏感性、图5-3 需中继点时空分布、图5-4 逐架次保障率
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
import os
BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, 'figs_精确')
os.makedirs(FIG, exist_ok=True)

relay2 = pd.read_csv(os.path.join(BASE, 'Q3_两架方案_中继排程.csv'))
relay3 = pd.read_csv(os.path.join(BASE, 'Q3_三架方案_中继排程.csv'))
per = pd.read_csv(os.path.join(BASE, 'Q3_两架方案_逐点通信验证.csv'))
per3 = pd.read_csv(os.path.join(BASE, 'Q3_三架方案_逐点通信验证.csv'))
R2 = json.load(open(os.path.join(BASE, 'Q3_两架方案_汇总.json'), encoding='utf-8'))
R3 = json.load(open(os.path.join(BASE, 'Q3_三架方案_汇总.json'), encoding='utf-8'))

# ---------- 图5-1 中继甘特图（两架，R01/R02双轨；R03增配对照用虚线示时间窗） ----------
fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150)
machines = ['R02', 'R01']
colors = {'R01': '#2e7dd1', 'R02': '#e63b2e', 'R03': '#2ea84f'}
labels = []
for r in relay2.iterrows():
    _, s = r
    m = s['中继编号']; y = machines.index(m)
    t0, t1, t2, t3, t4 = s['准备开始s'], s['到岗时刻s'], s['离岗时刻s'], s['返航落地s'], s['周转完成s']
    ax.add_patch(Rectangle((t0, y - 0.28), t1 - t0, 0.56, fc='#9aa5b1', ec='none'))
    ax.add_patch(Rectangle((t1, y - 0.28), t2 - t1, 0.56, fc=colors[m], ec='white', lw=0.8))
    ax.add_patch(Rectangle((t2, y - 0.28), t3 - t2, 0.56, fc='#f0a93b', ec='none'))
    ax.text((t0 + t1) / 2, y, '准备', ha='center', va='center', fontsize=8)
    ax.text((t1 + t2) / 2, y, s['任务ID'] + '\n(%d点)' % s['覆盖点数'], ha='center', va='center', fontsize=8,
            color='white', fontweight='bold')
    ax.text((t2 + t3) / 2, y, '返航', ha='center', va='center', fontsize=8)
    labels.append((m, s['任务ID']))
for s in relay3.iterrows():
    _, s = s
    if s['中继编号'] != 'R03':
        continue
    y = -1.0
    ax.add_patch(Rectangle((s['准备开始s'], y - 0.22), s['离岗时刻s'] - s['准备开始s'], 0.44,
                           fc=colors['R03'], ec='none', alpha=0.55))
    ax.text(s['准备开始s'] + 40, y, 'R03-%s' % s['任务ID'], fontsize=8, color=colors['R03'], va='center')
ax.set_yticks(list(range(2)))
ax.set_yticklabels(machines, fontsize=11)
ax.set_ylim(-1.4, 1.4)
ax.set_xlabel('时刻（s）', fontsize=11)
ax.set_title('图5-1 中继服务时段甘特图（两架主方案7架次；浅灰=准备、蓝/红=悬停服务、橙=返航；R03为增配对照）',
             fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
handles = [Line2D([0], [0], color='#9aa5b1', lw=6, label='准备/周转'),
           Line2D([0], [0], color='#2e7dd1', lw=6, label='R01悬停服务'),
           Line2D([0], [0], color='#e63b2e', lw=6, label='R02悬停服务'),
           Line2D([0], [0], color='#f0a93b', lw=6, label='返航/转场'),
           Line2D([0], [0], color='#2ea84f', lw=6, alpha=0.55, label='R03增配对照')]
ax.legend(handles=handles, loc='upper left', fontsize=9, ncol=5)
fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.14)
fig.savefig(os.path.join(FIG, 'Q3_图5-1_甘特图.png'))
plt.close(fig)
print('saved Q3_图5-1_甘特图.png')

# ---------- 图5-2 敏感性：(a)两架vs三架保障率/断链；(b)门限直连区数 ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
ax = axes[0]
scen = ['两架主方案', '增配三架']
rate = [R2['任务通信保障率'], R3['任务通信保障率']]
brk = [R2['断链'], R3['断链']]
x = np.arange(2)
b1 = ax.bar(x - 0.18, rate, 0.34, color='#2e7dd1', label='任务通信保障率(%)')
for xi, v in zip(x, rate):
    ax.text(xi - 0.18, v + 0.3, '%.2f' % v, ha='center', fontsize=9)
ax2 = ax.twinx()
b2 = ax2.bar(x + 0.18, brk, 0.34, color='#e63b2e', label='残余断链(点)')
for xi, v in zip(x, brk):
    ax2.text(xi + 0.18, v + 3, str(v), ha='center', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(scen, fontsize=11)
ax.set_ylim(90, 100); ax.set_ylabel('保障率(%)', fontsize=10)
ax2.set_ylabel('断链(点)', fontsize=10)
ax.set_title('(a) 中继数量敏感性（受限候选口径）', fontsize=12)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='lower left', fontsize=9)

ax = axes[1]
thr = ['-90 dBm', '-95 dBm', '-98 dBm']
direct = [3, 10, 15]
ax.bar(range(3), direct, 0.5, color='#2ea84f')
for i, v in enumerate(direct):
    ax.text(i, v + 0.2, '%d区' % v, ha='center', fontsize=10)
ax.set_xticks(range(3)); ax.set_xticklabels(thr, fontsize=11)
ax.set_ylabel('G01可直连服务区数', fontsize=10)
ax.set_title('(b) 接收门限敏感性', fontsize=12)
fig.suptitle('图5-2 中继数量与接收门限敏感性（200点/航段加密口径）', fontsize=13, fontweight='bold')
fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.12)
fig.savefig(os.path.join(FIG, 'Q3_图5-2_敏感性.png'))
plt.close(fig)
print('saved Q3_图5-2_敏感性.png')

# ---------- 图5-3 需中继点时空分布（时间×架次，绿=中继保障 红=断链） ----------
fig, ax = plt.subplots(figsize=(11, 5.2), dpi=150)
trip_order = sorted(per['架次'].unique(), key=lambda x: int(x[1:]))
pos = {t: i for i, t in enumerate(trip_order)}
covered = per[per['保障状态'] == '中继保障']
broken = per[per['保障状态'] == '断链']
ax.scatter(covered['t'], [pos[t] for t in covered['架次']], s=6, c='#2ea84f', alpha=0.7, label='中继保障点(%d)' % len(covered))
ax.scatter(broken['t'], [pos[t] for t in broken['架次']], s=10, c='#e63b2e', marker='x', label='断链点(%d)' % len(broken))
for s in relay2.iterrows():
    _, r = s
    ax.axvspan(r['到岗时刻s'], r['离岗时刻s'], color='#2e7dd1', alpha=0.10)
ax.set_yticks(range(len(trip_order)))
ax.set_yticklabels(trip_order, fontsize=9)
ax.set_xlabel('时刻（s）', fontsize=11)
ax.set_ylabel('运输架次', fontsize=11)
ax.set_title('图5-3 需中继点时空分布与中继服务时段（两架主方案，绿=中继保障 红=断链）',
             fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend(loc='upper left', fontsize=9)
fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.10)
fig.savefig(os.path.join(FIG, 'Q3_图5-3_时空分布.png'))
plt.close(fig)
print('saved Q3_图5-3_时空分布.png')

# ---------- 图5-4 逐架次通信保障率（两架 vs 三架对照） ----------
fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150)
def per_trip_rate(df):
    g = df.groupby('架次')['保障状态'].agg(lambda s: (s == '中继保障').sum() / len(s))
    return g
r2 = per_trip_rate(per)
r3 = per_trip_rate(per3)
idx = sorted(set(r2.index) | set(r3.index), key=lambda x: int(x[1:]))
x = np.arange(len(idx))
ax.bar(x - 0.2, [r2.get(t, 0) * 100 for t in idx], 0.38, color='#2e7dd1', label='两架主方案')
ax.bar(x + 0.2, [r3.get(t, 0) * 100 for t in idx], 0.38, color='#f0a93b', label='增配三架')
for xi, t in enumerate(idx):
    v2 = r2.get(t, 0) * 100
    if 0 < v2 < 100:
        ax.text(xi - 0.2, v2 + 0.5, '%.0f%%' % v2, ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(idx, fontsize=9)
ax.set_ylim(0, 108)
ax.set_ylabel('需中继点保障率(%)', fontsize=11)
ax.set_xlabel('运输架次（仅含需中继架次）', fontsize=11)
ax.set_title('图5-4 逐架次任务通信保障率（200点/航段加密口径）', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.12)
fig.savefig(os.path.join(FIG, 'Q3_图5-4_逐架次保障率.png'))
plt.close(fig)
print('saved Q3_图5-4_逐架次保障率.png')
print('Q3 4图完成')
