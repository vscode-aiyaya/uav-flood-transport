# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。AI工具名称：DeepSeek；版本/型号：V4.1-Flash；
# 开发机构：杭州深度求索人工智能基础技术研究有限公司；版本发布日期：2026年9月10日。

"""重绘图3-4/3-7/3-8/3-9（Q1复核清单第3条）"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

import os
BASE = os.path.dirname(os.path.abspath(__file__))
PREP = os.path.join(BASE, '预处理输出')
OUTD = os.path.join(BASE, 'figs_精确')

for f in [r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf']:
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=f).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False
os.makedirs(OUTD, exist_ok=True)

# ============ 图3-4 机队方案对比 ============
fleet = pd.read_csv(os.path.join(BASE, 'Q1_机队方案对比.csv'))
fmap = dict(zip(fleet['方案'], fleet[['总架次', '总能耗kWh']].values.tolist()))
names = ['全A机队', '全B机队', '全C机队', '混合机队']
trips = [fmap['全A'][0], fmap['全B'][0], fmap['全C'][0], fmap['混合'][0]]
ener = [fmap['全A'][1], fmap['全B'][1], fmap['全C'][1], fmap['混合'][1]]
fig, ax = plt.subplots(figsize=(8.2, 4.9), dpi=200)
x = np.arange(4)
bars = ax.bar(x, trips, 0.55, color=['#4C78A8', '#4C78A8', '#4C78A8', '#E45756'], label='总架次')
ax.set_ylabel('总架次数（次）', fontsize=12)
ax2 = ax.twinx()
ax2.plot(x, ener, 'o-', color='#2F4F4F', lw=2, ms=8, label='总能耗')
ax2.set_ylabel('总能耗（kWh）', fontsize=12)
ax2.set_ylim(0, 130)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=12)
ax.set_ylim(0, 62)
for i, (t, e) in enumerate(zip(trips, ener)):
    ax.annotate(f'{t}', (x[i], trips[i]), textcoords='offset points', xytext=(0, 4), ha='center', fontsize=11)
    ax2.annotate(f'{e:.1f}', (x[i], ener[i]), textcoords='offset points', xytext=(0, -16), ha='center', fontsize=11, color='#2F4F4F')
ax.set_title('图3-4  机队方案对比（各机型单独承担全部80箱 vs 混合18架次）', fontsize=13)
ax.grid(axis='y', alpha=0.3, ls='--')
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTD, 'fig3_4_机队对比.png'))
plt.close()

# ============ 图3-7 距离-能耗散点拟合 ============
trips_df = pd.read_csv(os.path.join(BASE, 'Q1_精确_组批方案.csv'))
nodes = pd.read_csv(os.path.join(PREP, 'Q1_节点_预处理.csv'))
dist = dict(zip(nodes['编号'], nodes['距O01水平距离(m)']))
trips_df['距离m'] = trips_df['服务区'].map(dist)
fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=200)
for mt, color, mk in [('B', '#4C78A8', 'o'), ('C', '#E45756', 's')]:
    sub = trips_df[trips_df['机型'] == mt]
    xx, yy = sub['距离m'].values, sub['能耗kWh'].values
    ax.scatter(xx, yy, color=color, marker=mk, s=55, label=f'{mt}型（{len(sub)}架次）', zorder=3)
    k, b = np.polyfit(xx, yy, 1)
    r2 = 1 - np.sum((yy - (k * xx + b)) ** 2) / np.sum((yy - yy.mean()) ** 2)
    xs = np.linspace(xx.min() - 200, xx.max() + 200, 50)
    ax.plot(xs, k * xs + b, '--', color=color, lw=1.6, alpha=0.8)
    ax.annotate(f'{mt}型 R²={r2:.3f}', xy=(xs[int(len(xs)*0.62)], k * xs[int(len(xs)*0.62)] + b),
                fontsize=11, color=color)
ax.set_xlabel('水平距离（m）', fontsize=12)
ax.set_ylabel('架次能耗（kWh）', fontsize=12)
ax.set_title('图3-7  距离-架次能耗散点及分机型线性拟合（18架次精确方案）', fontsize=13)
ax.grid(alpha=0.35, ls='--')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTD, 'fig3_7_能耗散点.png'))
plt.close()

# ============ 图3-8 能耗构成（精确重算） ============
import importlib.util, sys
spec = importlib.util.spec_from_file_location('q1', os.path.join(BASE, 'Q1_精确求解.py'))
mod = importlib.util.module_from_spec(spec); sys.modules['q1'] = mod
spec.loader.exec_module(mod)
pps, energy, Lg = mod.pps, mod.energy, mod.Lg
go_h = ret_h = climb_go = climb_ret = 0.0
tot = 0.0
for _, t in trips_df.iterrows():
    pp = pps[t['机型']]
    r = mod.sa_rows[mod.sa_rows['编号'] == t['服务区']].iloc[0]
    d = float(r['d']); H = float(t['巡航海拔m']); hsa = float(r['海拔(m)']); q = float(t['载重kg'])
    go = pp['E'] * d / Lg(pp, q); ret = pp['E'] * d / pp['L0']
    up1 = 9.81 * (pp['m0'] + q) * (H - mod.O_ALT) / (pp['eta'] * 3600000)
    up2 = 9.81 * pp['m0'] * (H - hsa - 30) / (pp['eta'] * 3600000)
    go_h += go; ret_h += ret; climb_go += up1; climb_ret += up2
    tot += go + ret + up1 + up2
labels = ['水平-去程', '水平-返程', '爬升-去程', '爬升-返程']
vals = [go_h, ret_h, climb_go, climb_ret]
print('图3-8 能耗构成: 总=%.4f' % tot, '占比: ' + ' '.join(f'{v/tot*100:.2f}%' for v in vals))
fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=200)
wedges, _, autot = ax.pie(vals, labels=labels, autopct=lambda p: f'{p:.1f}%',
                          colors=['#4C78A8', '#F2A900', '#E45756', '#A7BED3'],
                          startangle=90, counterclock=False, wedgeprops=dict(width=0.42, edgecolor='w'))
for t in autot:
    t.set_fontsize(11)
ax.set_title(f'图3-8  18架次方案总能耗构成（合计{tot:.2f} kWh）', fontsize=13)
ax.legend(wedges, [f'{l}  {v/tot*100:.1f}%' for l, v in zip(labels, vals)],
          loc='lower center', bbox_to_anchor=(0.5, -0.06), fontsize=10, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(OUTD, 'fig3_8_能耗构成.png'))
plt.close()

# ============ 图3-9 灵敏度 (a) + m_safe热力图 (b) ============
sens = pd.read_csv(os.path.join(BASE, 'Q1_精确_eta敏感性.csv'))
rhos = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
fig = plt.figure(figsize=(13.5, 5.6), dpi=200)
ax1 = fig.add_subplot(1, 2, 1)
xr = np.arange(len(rhos))
ts = [sens.loc[sens['返航余量'] == r, '最少架次数'].iloc[0] if not sens.loc[sens['返航余量'] == r, '可行'].iloc[0] == False else np.nan for r in rhos]
es = [sens.loc[sens['返航余量'] == r, '总能耗kWh'].iloc[0] if not sens.loc[sens['返航余量'] == r, '可行'].iloc[0] == False else np.nan for r in rhos]
hs = [sens.loc[sens['返航余量'] == r, '累计作业时间h'].iloc[0] if not sens.loc[sens['返航余量'] == r, '可行'].iloc[0] == False else np.nan for r in rhos]
ax1.bar(xr, ts, 0.5, color='#4C78A8', label='最少架次数')
ax1.set_ylabel('最少架次数（次）', fontsize=11)
ax1.set_ylim(0, 28)
ax1b = ax1.twinx()
ax1b.plot(xr, es, 'o-', color='#E45756', lw=2, ms=6, label='总能耗')
ax1b.plot(xr, hs, 's--', color='#2F4F4F', lw=2, ms=6, label='累计作业时间')
ax1b.set_ylabel('总能耗（kWh） / 累计作业时间（h）', fontsize=11)
ax1b.set_ylim(0, 90)
ax1.set_xticks(xr); ax1.set_xticklabels([f'{int(r*100)}%' for r in rhos], fontsize=10)
for i, (t, e, h) in enumerate(zip(ts, es, hs)):
    if not np.isnan(t):
        ax1.annotate(f'{int(t)}', (xr[i], t), textcoords='offset points', xytext=(0, 4), ha='center', fontsize=10)
        ax1b.annotate(f'{e:.1f}', (xr[i], e), textcoords='offset points', xytext=(-6, -14), fontsize=9, color='#E45756')
        ax1b.annotate(f'{h:.1f}', (xr[i], h), textcoords='offset points', xytext=(8, -14), fontsize=9, color='#2F4F4F')
ax1.annotate('40%不可行', xy=(6, 3), fontsize=11, color='#888888', ha='center')
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax1b.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=9.5)
ax1.set_title('(a) 安全余量对架次数、能耗和时间的影响', fontsize=12)
ax1.grid(axis='y', alpha=0.3, ls='--')

# (b) m_safe 热力图：三机型×15区×7档
sids = [f'S{i:03d}' for i in range(1, 16)]
def msafe_grid(mt, rhos):
    pp = pps[mt]; grid = np.full((15, len(rhos)), np.nan)
    for i, sid in enumerate(sids):
        r = mod.sa_rows[mod.sa_rows['编号'] == sid].iloc[0]
        d = float(r['d']); hsa = float(r['海拔(m)'])
        H = mod.cruise_exact(mod.O_LON, mod.O_LAT, float(r['经度']), float(r['纬度']))
        for j, rho in enumerate(rhos):
            if mod.energy(pp, 0.0, d, H, hsa) > (1 - rho) * pp['E']:
                grid[i, j] = np.nan
            else:
                lo, hi = 0.0, pp['Q']
                for _ in range(60):
                    mid = (lo + hi) / 2
                    if mod.energy(pp, mid, d, H, hsa) <= (1 - rho) * pp['E']:
                        lo = mid
                    else:
                        hi = mid
                grid[i, j] = lo
    return grid
gs = [msafe_grid(mt, rhos) for mt in ['A', 'B', 'C']]
gspec = fig.add_gridspec(1, 3, left=0.55, right=0.985, top=0.82, bottom=0.12, wspace=0.16)
for axb, mt, g in zip(gspec.subplots(), ['A', 'B', 'C'], gs):
    im = axb.imshow(g, aspect='auto', cmap='YlGnBu', vmin=0, vmax=pp['Q'] if (pp := pps[mt]) else 80)
    axb.set_xticks(range(len(rhos)))
    axb.set_xticklabels([f'{int(r*100)}%' for r in rhos], fontsize=8, rotation=0)
    axb.set_yticks(range(15)); axb.set_yticklabels(sids, fontsize=7)
    axb.set_title(f'{mt}型（Q={int(pps[mt]["Q"])} kg）', fontsize=10)
    for i in range(15):
        for j in range(7):
            v = g[i, j]
            if np.isnan(v):
                axb.text(j, i, '×', ha='center', va='center', fontsize=8, color='#B22222')
            else:
                axb.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=6.2, color='#222')
    cb = fig.colorbar(im, ax=axb, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)
fig.suptitle('图3-9  返航安全余量灵敏度分析（×=该机型空载往返不满足余量，航线不可达）', fontsize=13, y=0.97)
plt.savefig(os.path.join(OUTD, 'fig3_9_灵敏度.png'))
plt.close()
print('4张图已生成:', os.listdir(OUTD))
