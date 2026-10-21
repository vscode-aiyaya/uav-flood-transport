# -*- coding: utf-8 -*-
"""
Grayscale IEEE figures for the 10-page manuscript.

All three figures are drawn on a 2.6 in wide canvas, i.e. exactly their printed
width (0.74 of the 3.5 in IEEE column), so every font size below is
the physical size in the submitted PDF.

Schematic diagrams (fig 1/2): right-angle orthogonal paths only, no diagonal
arrows, no label on a line, a border or a marker.
Map panel (fig 3a): real geographic coordinates from the frozen data; transfer
segments are drawn as the straight direct-flight legs they physically are.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("FIG_OUT", _HERE)
_ANALYSIS = os.environ.get("FIG_ANALYSIS",
                           os.path.normpath(os.path.join(_HERE, "..", "analysis")))
REPO = _ANALYSIS
Q3B = os.environ.get("FIG_Q3OUT", os.path.join(_ANALYSIS, "q3_rebuild", "outputs"))

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "DejaVu Sans", "font.size": 7,
    "axes.linewidth": 0.6, "axes.labelsize": 6.2,
    "xtick.labelsize": 5.2, "ytick.labelsize": 5.2,
    "legend.fontsize": 5.0, "axes.unicode_minus": False,
})

BLACK = "#000000"; DGRAY = "#333333"; MGRAY = "#666666"
LGRAY = "#AAAAAA"; VLGRAY = "#D4D4D4"; WHITE = "#FFFFFF"
AHALF = 0.07
FIG_W = 2.60          # printed width in inches


def arrow_head(ax, x, y, direction, color=BLACK, lw=0.9, zorder=6):
    h = 0.12; w = 0.05   # elongated head (~2.4:1), not a flat triangle
    if direction == "right":   p = [(x, y), (x-h, y-w), (x-h, y+w)]
    elif direction == "left":  p = [(x, y), (x+h, y-w), (x+h, y+w)]
    elif direction == "up":    p = [(x, y), (x-w, y-h), (x+w, y-h)]
    elif direction == "down":  p = [(x, y), (x-w, y+h), (x+w, y+h)]
    else: return
    ax.add_patch(Polygon(p, fc=color, ec=color, lw=lw, closed=True, zorder=zorder))


def rect(ax, x, y, w, h, fc=WHITE, ec=BLACK, lw=1.0, ls="-"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))


def new_fig(xlim, ylim, fig_w=FIG_W):
    """Create fig+ax where ax exactly fills figure with given data limits.
    fig_height scales proportionally so no aspect distortion."""
    ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * ratio))
    ax.set_position([0, 0, 1, 1])  # ax occupies entire figure
    ax.axis("off")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    return fig, ax


# ============================================================ FIG 1
fig, ax = new_fig((0, 12.5), (0, 10.7))

# ---- shared geometry: 3 columns x 2 mirror-symmetric sub-boxes ----
BOX_W, BOX_H = 3.4, 2.25          # identical size for all six sub-boxes
CY = 4.9                          # vertical centre of the two-box stack
Y_TB = CY + 0.65                  # top sub-box bottom edge     -> 5.55
Y_BB = CY - 0.65 - BOX_H          # bottom sub-box bottom edge  -> 2.25
CY_T = Y_TB + BOX_H / 2           # 6.675
CY_B = Y_BB + BOX_H / 2           # 3.375
XL, XM, XR = 0.5, 4.45, 8.6       # left edge of each column
CX_L, CX_M, CX_R = XL + BOX_W / 2, XM + BOX_W / 2, XR + BOX_W / 2

# ---- left column: dispatch center ----
rect(ax, 0.25, 0.6, 3.9, 8.6, fc=WHITE, ec=BLACK, lw=1.2)
ax.text(CX_L, 8.42, "Dispatch center\nO01", ha="center", va="center",
        fontsize=5.6, weight="bold", linespacing=1.15)
rect(ax, XL, Y_TB, BOX_W, BOX_H, fc=WHITE, ec=BLACK, lw=0.9)
ax.text(CX_L, CY_T + 0.42, "Gateway G01", ha="center", va="center",
        fontsize=6.6, weight="bold")
ax.text(CX_L, CY_T - 0.40, "20 m AGL", ha="center", va="center",
        fontsize=5.6, color=MGRAY)
rect(ax, XL, Y_BB, BOX_W, BOX_H, fc=WHITE, ec=BLACK, lw=0.9)
ax.text(CX_L, CY_B + 0.42, "Battery pool", ha="center", va="center",
        fontsize=6.6, weight="bold")
ax.text(CX_L, CY_B - 0.34, "energy\ncomponents", ha="center", va="center",
        fontsize=5.4, color=MGRAY, linespacing=1.15)

# ---- middle column: transport UAVs over service areas ----
rect(ax, XM, Y_TB, BOX_W, BOX_H, fc=VLGRAY, ec=BLACK, lw=1.0)
ax.text(CX_M, CY_T + 0.34, "Transport\nUAVs", ha="center", va="center",
        fontsize=6.6, weight="bold", linespacing=1.15)
ax.text(CX_M, CY_T - 0.48, "Type A / B / C", ha="center", va="center",
        fontsize=5.4, color=DGRAY)
rect(ax, XM, Y_BB, BOX_W, BOX_H, fc=LGRAY, ec=BLACK, lw=1.0, ls="--")
ax.text(CX_M, CY_B + 0.42, "15 areas", ha="center", va="center",
        fontsize=6.4, weight="bold")
ax.text(CX_M, CY_B - 0.40, "SOC margin 20%", ha="center", va="center",
        fontsize=5.4, color=DGRAY)

# round-trip vertical arrow (middle column)
ax.plot([CX_M, CX_M], [Y_BB + BOX_H, Y_TB], color=DGRAY, lw=1.0,
        solid_capstyle="round", zorder=5)
arrow_head(ax, CX_M, Y_BB + BOX_H, "down", color=DGRAY)
arrow_head(ax, CX_M, Y_TB, "up", color=DGRAY)
ax.annotate("round trip", xy=(CX_M, CY), xytext=(-8, 0),
            xycoords="data", textcoords="offset points",
            fontsize=5.8, color=DGRAY, ha="right", va="center")

# ---- right column: one relay sortie in the shadowed back-slope zone ----
rect(ax, 8.4, 0.6, 3.8, 8.6, fc=WHITE, ec=MGRAY, lw=0.9, ls="--")
ax.text(CX_R, 8.45, "One relay sortie\n(terrain-shadowed)", ha="center", va="center",
        fontsize=5.4, color=DGRAY, linespacing=1.2)
rect(ax, XR, Y_TB, BOX_W, BOX_H, fc=DGRAY, ec=BLACK, lw=0.9)
ax.text(CX_R, CY_T, "hover point 1", ha="center", va="center",
        fontsize=5.8, color=WHITE, weight="bold")
rect(ax, XR, Y_BB, BOX_W, BOX_H, fc=DGRAY, ec=BLACK, lw=0.9)
ax.text(CX_R, CY_B, "hover point 2", ha="center", va="center",
        fontsize=5.8, color=WHITE, weight="bold")

# multi-hover transfer of the SAME airframe: hover point 1 -> hover point 2
ax.plot([CX_R, CX_R], [Y_BB + BOX_H, Y_TB], color=MGRAY, lw=1.0,
        solid_capstyle="round", zorder=5)
arrow_head(ax, CX_R, Y_BB + BOX_H, "down", color=MGRAY)
ax.text(9.32, CY, "direct\ntransfer", ha="center", va="center",
        fontsize=6.0, color=DGRAY, linespacing=1.2)

# ---- inter-column connectors (horizontal / vertical only) ----
ax.plot([XM + BOX_W, XR], [CY_T, CY_T], color=DGRAY, lw=1.0,
        solid_capstyle="round", zorder=5)
arrow_head(ax, XR, CY_T, "right", color=DGRAY)

# return-to-base swap: from hover point 2 down into the free lane below all
# sub-boxes, across to the dispatch column, then up into the battery pool
ax.plot([CX_R, CX_R], [Y_BB, 1.4], color=MGRAY, lw=0.8, ls="--",
        solid_capstyle="round", zorder=5)
ax.plot([CX_R, XL + BOX_W], [1.4, 1.4], color=MGRAY, lw=0.8, ls="--",
        solid_capstyle="round", zorder=5)
ax.plot([XL + BOX_W, XL + BOX_W], [1.4, Y_BB], color=MGRAY, lw=0.8, ls="--",
        solid_capstyle="round", zorder=5)
arrow_head(ax, CX_R, Y_BB, "up", color=MGRAY, lw=0.8)
arrow_head(ax, XL + BOX_W, Y_BB, "up", color=MGRAY, lw=0.8)
ax.text(6.28, 0.95, "return-to-base swap", ha="center", va="center",
        fontsize=5.4, color=MGRAY)

# two-hop note at top
ax.text(6.25, 10.4, "two-hop link: transport → relay → G01",
        ha="center", va="center", fontsize=6.6, color=DGRAY, weight="bold")
ax.text(6.25, 9.8, "(both directions ≥ −90 dBm)",
        ha="center", va="center", fontsize=5.6, color=MGRAY)

fig.savefig(os.path.join(OUT, "fig_system.pdf"))
fig.savefig(os.path.join(OUT, "fig_system.png"), dpi=300)
plt.close(fig)


# ============================================================ FIG 2
# ylim starts just below the lowest box so the caption hugs the figure
fig, ax = new_fig((0, 10.5), (1.95, 12.8))

# top shared physical-model band
rect(ax, 0.35, 11.05, 9.8, 1.35, fc=VLGRAY, ec=BLACK, lw=1.0, ls="--")
ax.text(5.25, 12.00, "Shared physical model (all four stages):",
        ha="center", fontsize=6.5, weight="bold")
ax.text(5.25, 11.45, "DEM cruise altitude | nonlinear range | two-stage charging",
        ha="center", fontsize=5.0, color=DGRAY)

# four problem boxes vertically stacked
BOX_W, BOX_H = 9.8, 1.55
BX = 0.35
CX = BX + BOX_W / 2
YS = [9.2, 6.9, 4.6, 2.25]
Heads = [
    "Problem 1: DEM-ELBPP",
    "Problem 2: HF-VRPBT",
    "Problem 3: MHPRS",
    "Problem 4: SC-MAE-RG",
]
Descs = [
    ["energy-aware bin packing"],
    ["heterogeneous-fleet routing", "with time windows"],
    ["multi-hover-point relay scheduling", "(fixed transport plan)"],
    ["clustering / resource accounting", "/ gap analysis"],
]
for h, d, y in zip(Heads, Descs, YS):
    rect(ax, BX, y, BOX_W, BOX_H, fc=WHITE, ec=BLACK, lw=1.0)
    ax.text(CX, y + BOX_H - 0.33, h, ha="center", va="center",
            fontsize=6.6, weight="bold")
    ax.text(CX, y + BOX_H - 0.70, "\n".join(d), ha="center", va="top",
            fontsize=5.5, linespacing=1.15)

# vertical ortho arrows between boxes + inherited artefact labels
Outputs = ["batches", "transport schedule", "transport + relay plan"]
for i in range(len(Heads) - 1):
    y_top = YS[i]
    y_bot = YS[i+1] + BOX_H
    ax.plot([CX, CX], [y_top, y_bot], color=BLACK, lw=1.0,
            solid_capstyle="round", zorder=5)
    arrow_head(ax, CX, y_bot, "down", color=BLACK)
    ax.annotate(Outputs[i], xy=(CX, (y_top + y_bot) / 2), xytext=(6, 0),
                xycoords="data", textcoords="offset points",
                fontsize=5.2, color=DGRAY, ha="left", va="center")

# side bracket: the shared model applies to all four stages
ax.plot([0.16, 0.16], [YS[-1], YS[0] + BOX_H], color=MGRAY, lw=0.8, ls="--",
        zorder=4)
ax.plot([0.16, 0.32], [YS[-1], YS[-1]], color=MGRAY, lw=0.8, ls="--", zorder=4)
ax.plot([0.16, 0.32], [YS[0] + BOX_H, YS[0] + BOX_H], color=MGRAY, lw=0.8,
        ls="--", zorder=4)

fig.savefig(os.path.join(OUT, "fig_route.pdf"))
fig.savefig(os.path.join(OUT, "fig_route.png"), dpi=300)
plt.close(fig)


# ============================================================ FIG 3
fig = plt.figure(figsize=(FIG_W, 3.21))
# generous side margins: left for y-tick labels, right for the last x-label
axm = fig.add_axes([0.185, 0.595, 0.795, 0.305])   # top (a)
axb = fig.add_axes([0.185, 0.155, 0.795, 0.295])   # bottom (b)

# ---------- (a) real geographic coordinates ----------
axm.set_xlim(109.155, 109.310); axm.set_ylim(22.980, 23.115)
axm.set_xticks([109.16, 109.20, 109.24, 109.28])
axm.set_yticks([23.00, 23.04, 23.08])

# real 15 service areas and the real O01/G01 location (frozen node table)
nodes = pd.read_csv(os.path.join(REPO, "预处理输出", "Q1_节点_预处理.csv"))
o01 = nodes[nodes["编号"] == "O01"].iloc[0]
O_lon, O_lat = float(o01["经度"]), float(o01["纬度"])
svc = nodes[nodes["编号"].astype(str).str.startswith("S")]
axm.scatter(svc["经度"].astype(float), svc["纬度"].astype(float),
            s=7, c=LGRAY, marker="o", zorder=2, alpha=0.85)
axm.scatter([O_lon], [O_lat], s=95, c=MGRAY, marker="*", zorder=4)
axm.annotate("O01 / G01", xy=(O_lon, O_lat), xytext=(0, -9),
             xycoords="data", textcoords="offset points",
             fontsize=5.4, color=MGRAY, ha="center", va="top")

# hover-point chains of the two multi-hover sorties, read from the unified
# frozen plan (script-02 verification summary; consecutive duplicates removed,
# since R01M1 serves its first two windows from the same physical point)
plan = pd.read_csv(os.path.join(Q3B, "Q3_重建_冻结方案核验汇总.csv"),
                   encoding="utf-8-sig")
def chain_points(mid):
    g = plan[plan["任务ID"] == mid].sort_values("时段")
    pts = []
    for _, r in g.iterrows():
        p = (float(r["悬停经度"]), float(r["悬停纬度"]))
        if not pts or abs(p[0] - pts[-1][0]) > 1e-9 or abs(p[1] - pts[-1][1]) > 1e-9:
            pts.append(p)
    return pts
c1 = chain_points("R01M1")   # [PER1_E, PER3-NE]
c2 = chain_points("R02M2")   # [PER3-SW, PER4]

# Chain 1: straight direct-flight transfer leg, PER1_E -> PER3-NE
(p1x, p1y), (p2x, p2y) = c1[0], c1[1]
axm.plot([p1x, p2x], [p1y, p2y], color=BLACK, lw=0.9, ls="--",
         solid_capstyle="round", zorder=5)
axm.scatter([p1x], [p1y], s=62, c=WHITE, edgecolors=BLACK, marker="s", zorder=7, linewidths=1.1)
axm.scatter([p2x], [p2y], s=62, c=WHITE, edgecolors=BLACK, marker="s", zorder=7, linewidths=1.1)
axm.annotate("PER1_E", xy=(p1x, p1y), xytext=(4, 2), xycoords="data",
             textcoords="offset points", fontsize=5.0, ha="left", va="bottom")
axm.annotate("PER3-NE", xy=(p2x, p2y), xytext=(6, 9), xycoords="data",
             textcoords="offset points", fontsize=5.3, ha="left", va="bottom")

# Chain 2: straight direct-flight transfer leg, PER3-SW -> PER4
(q1x, q1y), (q2x, q2y) = c2[0], c2[1]
axm.plot([q1x, q2x], [q1y, q2y], color=DGRAY, lw=0.9, ls=(0, (1.2, 1.0)),
         solid_capstyle="round", zorder=5)
axm.scatter([q1x], [q1y], s=62, c=WHITE, edgecolors=DGRAY, marker="^", zorder=7, linewidths=1.1)
axm.scatter([q2x], [q2y], s=62, c=WHITE, edgecolors=DGRAY, marker="^", zorder=7, linewidths=1.1)
axm.annotate("PER3-SW", xy=(q1x, q1y), xytext=(-15, -2), xycoords="data",
             textcoords="offset points", fontsize=5.3, color=DGRAY, ha="right", va="center")
axm.annotate("PER4", xy=(q2x, q2y), xytext=(-5, 2), xycoords="data",
             textcoords="offset points", fontsize=5.3, color=DGRAY, ha="right", va="bottom")

# chain identifiers go into an axes legend placed in the empty upper-left
# corner, so no text sits on a link line, a marker or the frame
axm.legend(handles=[
    Line2D([0], [0], color=BLACK, lw=0.9, ls="--", marker="s",
           markersize=3.6, markerfacecolor=WHITE, markeredgecolor=BLACK,
           markeredgewidth=0.7, label="Chain 1"),
    Line2D([0], [0], color=DGRAY, lw=0.9, ls=(0, (1.2, 1.0)), marker="^",
           markersize=3.6, markerfacecolor=WHITE, markeredgecolor=DGRAY,
           markeredgewidth=0.7, label="Chain 2")],
    loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=5.0,
    frameon=False, handlelength=2.0, handletextpad=0.4,
    labelspacing=0.3, borderpad=0.0, borderaxespad=0.4)

axm.set_xlabel("longitude (deg)", fontsize=6.0, labelpad=2)
axm.set_ylabel("latitude (deg)", fontsize=6.0, labelpad=2)
# panel labels only; descriptions live in the LaTeX caption (IEEE style)
axm.set_title("(a)", fontsize=6.2, pad=3)
axm.tick_params(labelsize=5.0, pad=1.5)

# ---------- (b) ----------
# bars read from computed outputs: script-04 single-hover baseline on the
# canonical demand set, and the script-06 MHPRS verification (11713/11713)
xpos = np.arange(4)
_base = pd.read_csv(os.path.join(Q3B, "Q3_重建_单悬停基线_保障率.csv"),
                    encoding="utf-8-sig")
vals = [round(100.0 * float(_base.loc[_base["中继机架数"] == n, "保障需中继点"].iloc[0])
              / float(_base.loc[_base["中继机架数"] == n, "总需中继点"].iloc[0]), 1)
        for n in (1, 2, 3)] + [100.0]
axb.bar(xpos[:3], vals[:3], width=0.58, fc=VLGRAY, ec=BLACK, lw=0.7,
        hatch="//", label="Single-hover", zorder=3)
axb.bar([xpos[3]], [vals[3]], width=0.58, fc=DGRAY, ec=BLACK, lw=0.7,
        hatch="xx", label="MHPRS (this work)", zorder=3)
for xi, v in zip(xpos[:3], vals[:3]):
    axb.annotate("%.1f" % v, xy=(xi, v), xytext=(0, 2.5),
                 xycoords="data", textcoords="offset points",
                 fontsize=5.2, ha="center", va="bottom", color=DGRAY)
# MHPRS value above its bar (never inside: keeps the label off the hatch)
axb.annotate("100.0", xy=(xpos[3], vals[3]), xytext=(0, 2.5),
             xycoords="data", textcoords="offset points",
             fontsize=5.2, ha="center", va="bottom", color=BLACK, weight="bold")
axb.set_xticks(xpos)
axb.set_xticklabels(["1 airframe", "2 airframes", "3 airframes",
                     "MHPRS\n(2 airframes)"], fontsize=5.0, linespacing=1.15)
axb.set_ylim(0, 130)
axb.set_yticks([0, 20, 40, 60, 80, 100])
axb.set_ylabel("relay-required points\ncovered (%)", fontsize=6.0, labelpad=2)
axb.set_title("(b)", fontsize=6.2, pad=3)
axb.tick_params(labelsize=5.0, pad=1.5)
# legend: placed well below the two-line x-tick labels, never over any data
axb.legend(loc="lower center", bbox_to_anchor=(0.5, -0.46), ncol=2,
           fontsize=4.8, framealpha=0.95, borderpad=0.35,
           handlelength=1.7, labelspacing=0.25, columnspacing=1.0)
for s in ("top", "right"):
    axb.spines[s].set_visible(False)

fig.savefig(os.path.join(OUT, "fig_mhprs.pdf"))
fig.savefig(os.path.join(OUT, "fig_mhprs.png"), dpi=300)
plt.close(fig)

print("All figures generated successfully.")
