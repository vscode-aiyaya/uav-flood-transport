"""Three IEEE single-column grayscale figures. Ortho L/Z paths only.
No diagonal arrows. Legend never overlays data.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.linewidth": 0.7, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.2, "axes.unicode_minus": False,
})

BLACK = "#000000"; DGRAY = "#333333"; MGRAY = "#666666"
LGRAY = "#AAAAAA"; VLGRAY = "#D4D4D4"; WHITE = "#FFFFFF"
AHALF = 0.07


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


def ortho_L(ax, x1, y1, x2, y2, color=BLACK, lw=1.0, ls="-", zorder=5):
    ax.plot([x1, x2], [y1, y1], color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=zorder)
    ax.plot([x2, x2], [y1, y2], color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=zorder)


def new_fig(xlim, ylim, fig_w=3.4):
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
ax.text(CX_L, 8.45, "Dispatch center O01", ha="center", va="center",
        fontsize=6.4, weight="bold")
rect(ax, XL, Y_TB, BOX_W, BOX_H, fc=WHITE, ec=BLACK, lw=0.9)
ax.text(CX_L, CY_T + 0.42, "Gateway G01", ha="center", va="center",
        fontsize=7.0, weight="bold")
ax.text(CX_L, CY_T - 0.40, "20 m above ground", ha="center", va="center",
        fontsize=5.4, color=MGRAY)
rect(ax, XL, Y_BB, BOX_W, BOX_H, fc=WHITE, ec=BLACK, lw=0.9)
ax.text(CX_L, CY_B + 0.42, "Battery pool", ha="center", va="center",
        fontsize=7.0, weight="bold")
ax.text(CX_L, CY_B - 0.40, "(energy components)", ha="center", va="center",
        fontsize=5.4, color=MGRAY)

# ---- middle column: transport UAVs over service areas ----
rect(ax, XM, Y_TB, BOX_W, BOX_H, fc=VLGRAY, ec=BLACK, lw=1.0)
ax.text(CX_M, CY_T + 0.42, "Transport UAVs", ha="center", va="center",
        fontsize=6.5, weight="bold")
ax.text(CX_M, CY_T - 0.40, "Type A / B / C", ha="center", va="center",
        fontsize=6.2, color=DGRAY)
rect(ax, XM, Y_BB, BOX_W, BOX_H, fc=LGRAY, ec=BLACK, lw=1.0, ls="--")
ax.text(CX_M, CY_B + 0.42, "15 service areas", ha="center", va="center",
        fontsize=6.5, weight="bold")
ax.text(CX_M, CY_B - 0.40, "return margin 20%", ha="center", va="center",
        fontsize=6.2, color=DGRAY)

# round-trip vertical arrow (middle column)
ax.plot([CX_M, CX_M], [Y_BB + BOX_H, Y_TB], color=DGRAY, lw=1.0,
        solid_capstyle="round", zorder=5)
arrow_head(ax, CX_M, Y_BB + BOX_H, "down", color=DGRAY)
arrow_head(ax, CX_M, Y_TB, "up", color=DGRAY)
ax.annotate("round trip", xy=(CX_M, CY), xytext=(8, 0),
            xycoords="data", textcoords="offset points",
            fontsize=6.0, color=DGRAY, ha="left", va="center")

# ---- right column: shadowed back-slope relay zone ----
rect(ax, 8.4, 0.6, 3.8, 8.6, fc=WHITE, ec=MGRAY, lw=0.9, ls="--")
ax.text(CX_R, 8.45, "Shadowed back-slope", ha="center", va="center",
        fontsize=6.2, color=DGRAY)
rect(ax, XR, Y_TB, BOX_W, BOX_H, fc=DGRAY, ec=BLACK, lw=0.9)
ax.text(CX_R, CY_T, "Relay UAV 1", ha="center", va="center",
        fontsize=7.0, color=WHITE, weight="bold")
rect(ax, XR, Y_BB, BOX_W, BOX_H, fc=DGRAY, ec=BLACK, lw=0.9)
ax.text(CX_R, CY_B, "Relay UAV 2", ha="center", va="center",
        fontsize=7.0, color=WHITE, weight="bold")

# multi-hover vertical arrow (right column)
ax.plot([CX_R, CX_R], [Y_BB + BOX_H, Y_TB], color=MGRAY, lw=1.0,
        solid_capstyle="round", zorder=5)
arrow_head(ax, CX_R, Y_BB + BOX_H, "down", color=MGRAY)
arrow_head(ax, CX_R, Y_TB, "up", color=MGRAY)
ax.text(9.30, CY, "multi-hover\ndirect\ntransfer", ha="center", va="center",
        fontsize=5.1, color=DGRAY, linespacing=1.3)

# ---- inter-column connectors (horizontal / vertical only) ----
ax.plot([XM + BOX_W, XR], [CY_T, CY_T], color=DGRAY, lw=1.0,
        solid_capstyle="round", zorder=5)
arrow_head(ax, XR, CY_T, "right", color=DGRAY)

# return-to-base swap, routed in the free lane below all sub-boxes
ax.plot([XR, XL + BOX_W], [1.4, 1.4], color=MGRAY, lw=0.8, ls="--",
        solid_capstyle="round", zorder=5)
arrow_head(ax, XL + BOX_W, 1.4, "left", color=MGRAY, lw=0.8)
arrow_head(ax, XR, 1.4, "right", color=MGRAY, lw=0.8)
ax.text(6.28, 0.95, "return-to-base swap", ha="center", va="center",
        fontsize=5.6, color=MGRAY)

# two-hop note at top: even 0.6-unit ladder from the frame-top row (9.2)
ax.text(6.25, 10.4, "two-hop link: transport → relay → G01",
        ha="center", va="center", fontsize=6.2, color=DGRAY, weight="bold")
ax.text(6.25, 9.8, "(both directions ≥ −90 dBm)",
        ha="center", va="center", fontsize=5.6, color=MGRAY)

fig.savefig(os.path.join(OUT, "fig_system.pdf"))
fig.savefig(os.path.join(OUT, "fig_system.png"), dpi=220)
plt.close(fig)


# ============================================================ FIG 2
# ylim starts just below the lowest box so the caption hugs the figure
fig, ax = new_fig((0, 10.5), (1.95, 12.8))

# top shared caliber band
rect(ax, 0.35, 11.1, 9.8, 1.25, fc=VLGRAY, ec=BLACK, lw=1.0, ls="--")
ax.text(5.25, 12.05, "Shared physical caliber (all four stages):",
        ha="center", fontsize=6.4, weight="bold")
ax.text(5.25, 11.5, "DEM cruise altitude | nonlinear equivalent range | two-stage charging",
        ha="center", fontsize=5.8, color=DGRAY)

# four problem boxes vertically stacked
BOX_W, BOX_H = 9.8, 1.45
BX = 0.35
YS = [9.1, 6.85, 4.55, 2.25]
Labs = [
    "Problem 1: DEM-ELBPP\nenergy-aware bin packing",
    "Problem 2: HF-VRPBT\nheterogeneous-fleet routing with time windows",
    "Problem 3: TD-CJO\njoint transport-communication scheduling (MHPRS)",
    "Problem 4: SC-MAE-RG\nclustering / resource accounting / gap analysis",
]
for s, y in zip(Labs, YS):
    rect(ax, BX, y, BOX_W, BOX_H, fc=WHITE, ec=BLACK, lw=1.0)
    ax.text(BX + BOX_W/2, y + BOX_H/2, s, ha="center", va="center",
            fontsize=6.8, weight="bold", linespacing=1.25)

# vertical ortho arrows between boxes + "reuse" labels
CX = BX + BOX_W/2
for i in range(len(Labs) - 1):
    y_top = YS[i]
    y_bot = YS[i+1] + BOX_H
    ax.plot([CX, CX], [y_top, y_bot], color=BLACK, lw=1.0, solid_capstyle="round", zorder=5)
    arrow_head(ax, CX, y_bot, "down", color=BLACK)
    ax.annotate("reuse", xy=(CX, (y_top + y_bot) / 2), xytext=(8, 0),
                xycoords="data", textcoords="offset points",
                fontsize=6.0, color=DGRAY, ha="left", va="center")

fig.savefig(os.path.join(OUT, "fig_route.pdf"))
fig.savefig(os.path.join(OUT, "fig_route.png"), dpi=220)
plt.close(fig)


# ============================================================ FIG 3
fig = plt.figure(figsize=(3.4, 4.2))
# generous side margins: left for y-tick labels, right for the last x-label
axm = fig.add_axes([0.17, 0.585, 0.80, 0.315])   # top (a)
axb = fig.add_axes([0.17, 0.135, 0.80, 0.320])   # bottom (b)

# ---------- (a) ----------
axm.set_xlim(109.155, 109.310); axm.set_ylim(22.980, 23.115)
rng = np.random.default_rng(7)
sx = 109.17 + rng.random(15) * 0.12
sy = 23.00 + rng.random(15) * 0.09
axm.scatter(sx, sy, s=9, c=LGRAY, marker="o", zorder=2, alpha=0.8)

axm.scatter([109.238], [23.048], s=130, c=MGRAY, marker="*", zorder=4)
axm.annotate("O01 / G01", xy=(109.238, 23.048), xytext=(-16, 14),
             xycoords="data", textcoords="offset points",
             fontsize=6.6, color=MGRAY, ha="right", va="bottom")

# Chain 1: horizontal-first L, PER1 -> PER3-NE
p1x, p1y, p1n = 109.28168, 23.02521, "PER1"
p2x, p2y, p2n = 109.18278, 23.08139, "PER3-NE"
ortho_L(axm, p1x, p1y, p2x, p1y, color=BLACK, lw=1.1, ls="--")
axm.plot([p2x, p2x], [p1y, p2y], color=BLACK, lw=1.1, ls="--", solid_capstyle="round", zorder=5)
axm.scatter([p1x], [p1y], s=85, c=WHITE, edgecolors=BLACK, marker="s", zorder=7, linewidths=1.3)
axm.scatter([p2x], [p2y], s=85, c=WHITE, edgecolors=BLACK, marker="s", zorder=7, linewidths=1.3)
axm.annotate(p1n, xy=(p1x, p1y), xytext=(10, -16), xycoords="data",
             textcoords="offset points", fontsize=6.3, ha="left", va="top")
axm.annotate(p2n, xy=(p2x, p2y), xytext=(6, 12), xycoords="data",
             textcoords="offset points", fontsize=6.3, ha="left", va="bottom")
# Chain 1 label shifted right so it clears the Chain-2 label and PER1 marker
axm.annotate("Chain 1 (R01M1)", xy=(109.262, p1y),
             xytext=(0, -10), xycoords="data", textcoords="offset points",
             fontsize=5.6, ha="center", va="top", style="italic")

# Chain 2: horizontal-first L, PER3-SW -> PER4
q1x, q1y, q1n = 109.21286, 23.01971, "PER3-SW"
q2x, q2y, q2n = 109.20278, 23.03139, "PER4"
ortho_L(axm, q1x, q1y, q2x, q1y, color=DGRAY, lw=1.1, ls=(0, (1.2, 1.0)))
axm.plot([q2x, q2x], [q1y, q2y], color=DGRAY, lw=1.1, ls=(0, (1.2, 1.0)), solid_capstyle="round", zorder=5)
axm.scatter([q1x], [q1y], s=85, c=WHITE, edgecolors=DGRAY, marker="^", zorder=7, linewidths=1.3)
axm.scatter([q2x], [q2y], s=85, c=WHITE, edgecolors=DGRAY, marker="^", zorder=7, linewidths=1.3)
# PER3-SW label moved to the left of its marker at mid-height, clearing Chain-2 label
axm.annotate(q1n, xy=(q1x, q1y), xytext=(-17, -2), xycoords="data",
             textcoords="offset points", fontsize=6.3, color=DGRAY, ha="right", va="center")
axm.annotate(q2n, xy=(q2x, q2y), xytext=(14, 10), xycoords="data",
             textcoords="offset points", fontsize=6.3, color=DGRAY, ha="left", va="bottom")
axm.annotate("Chain 2 (R02M2)", xy=((q1x + q2x) / 2, q1y),
             xytext=(0, -13), xycoords="data", textcoords="offset points",
             fontsize=5.4, color=DGRAY, ha="center", va="top", style="italic")

axm.set_xlabel("longitude (deg)", fontsize=6.6, labelpad=2)
axm.set_ylabel("latitude (deg)", fontsize=6.6, labelpad=2)
# panel labels only; descriptions live in the LaTeX caption (IEEE style)
axm.set_title("(a)", fontsize=6.4, pad=4)
axm.tick_params(labelsize=5.6, pad=1.5)

# ---------- (b) ----------
# one bar per configuration: baseline (1/2/3 airframes) then MHPRS last, so the
# MHPRS bar is never interleaved between the baseline categories
xpos = np.arange(4)
vals = [51.0, 62.7, 82.1, 100.0]
axb.bar(xpos[:3], vals[:3], width=0.58, fc=VLGRAY, ec=BLACK, lw=0.8,
        hatch="///", label="Single-hover round-trip baseline", zorder=3)
axb.bar([xpos[3]], [vals[3]], width=0.58, fc=DGRAY, ec=BLACK, lw=0.8,
        hatch="xxx", label="MHPRS (this work)", zorder=3)
for xi, v in zip(xpos[:3], vals[:3]):
    axb.annotate("%.1f" % v, xy=(xi, v), xytext=(0, 3),
                 xycoords="data", textcoords="offset points",
                 fontsize=5.8, ha="center", va="bottom", color=DGRAY)
# MHPRS value ABOVE its bar (never inside: keeps the label off the hatch)
axb.annotate("100\n(11683/11683)", xy=(xpos[3], vals[3]), xytext=(0, 3),
             xycoords="data", textcoords="offset points",
             fontsize=5.4, ha="center", va="bottom", color=BLACK,
             weight="bold", linespacing=1.35)
axb.set_xticks(xpos)
axb.set_xticklabels(["1 airframe", "2 airframes", "3 airframes", "MHPRS"],
                    fontsize=6.0)
axb.set_ylim(0, 130)
axb.set_yticks([0, 20, 40, 60, 80, 100])
axb.set_ylabel("relay-required points\ncovered (%)", fontsize=6.2, labelpad=2)
axb.set_title("(b)", fontsize=6.4, pad=4)
axb.tick_params(labelsize=5.8, pad=1.5)
# legend: placed well below the x-tick labels (never covers them or any data)
axb.legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=2,
           fontsize=5.2, framealpha=0.95, borderpad=0.4,
           handlelength=1.8, labelspacing=0.3, columnspacing=1.2)
for s in ("top", "right"):
    axb.spines[s].set_visible(False)

fig.savefig(os.path.join(OUT, "fig_mhprs.pdf"))
fig.savefig(os.path.join(OUT, "fig_mhprs.png"), dpi=220)
plt.close(fig)

print("All figures generated successfully.")
