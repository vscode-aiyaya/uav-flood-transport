# Multi-Hover-Point Relay Scheduling for UAV Emergency Delivery under a Fixed Transport Plan

Reproducibility repository accompanying the manuscript:

> **Multi-Hover-Point Relay Scheduling for UAV Emergency Delivery under a
> Fixed Transport Plan**
> Yang Xin, Zhao Zidong, Song Yuwei.
> *Manuscript, 2026* (preprint/journal reference to be added).

The repository contains the full four-stage model chain — exact energy-aware
bin packing (Problem 1), heterogeneous routing with battery turnover
(Problem 2), multi-hover-point relay scheduling / MHPRS (Problem 3), and
partition-based resource accounting / SC-MAE-RG (Problem 4) — together with
the clustering benchmark added in the journal version.

All reported numbers are produced by the scripts in this repository from the
shipped data; nothing is hard-coded from the contest solution.

## Repository layout

```
.
├── analysis/
│   ├── 数据/                    # raw instance: 30 m DEM (.mat) + contest .xlsx tables
│   ├── 预处理输出/              # 13 preprocessed CSV tables
│   ├── Q1_精确求解.py           # P1 exact set-partitioning batching (DP)
│   ├── Q1_重绘图3_4_7_8_9.py
│   ├── Q2_精确调度.py           # P2 candidate enumeration + urgency-first scheduling
│   ├── Q2_约束校验.py / Q2_输出生成.py / Q2_四指标非支配.py
│   ├── Q3_TD-CJO_v2.py          # P3 MHPRS plan construction
│   ├── Q3_复核.py               # P3 independent 1 s point-by-point verification
│   ├── Q3_绘图_v2.py
│   ├── q3_rebuild/              # P3 independent first-principles rebuild (canonical set)
│   │   ├── 01_轨迹与需中继点.py    # trajectory + relay-required points (11 713)
│   │   ├── 02_冻结方案独立核验.py  # frozen plan + missing hover-point reconstruction
│   │   ├── 04_单悬停基线.py
│   │   ├── 05_门限敏感性.py
│   │   ├── 06_规范集合核验.py      # coverage on the canonical 11 713-point set
│   │   ├── 07_能耗SOC_核验方案.py  # energy/SOC under the prior eta_c = 0.72
│   │   ├── 08_固定方案可行性核验.py
│   │   ├── 09_下界_中继机数.py     # >= 2 relay-airframe lower bound (Observation 1)
│   │   ├── 10_只读验证器.py        # read-only feasibility + clearance-convention audit
│   │   └── outputs/             # canonical artifacts of the single chain 01->02->06->07->08
│   ├── Q4_分区核算_v3.py        # P4 SC-MAE-RG resource accounting
│   ├── Q4_TOPSIS.py / Q4_绘图_v3.py
│   ├── *.csv / *.json           # frozen plans, logs and audit artifacts
│   └── baselines/               # external benchmark (journal version)
│       ├── q4_cluster_baseline.py  # K-means / Ward vs SC-MAE-RG
│       └── *.json / *.csv       # benchmark outputs
├── figures/
│   ├── make_figs.py             # grayscale IEEE figures
│   └── fig_system.pdf, fig_route.pdf, fig_mhprs.pdf
├── manuscript/
│   └── main_en.tex              # IEEE journal manuscript (10 pages)
├── requirements.txt
└── LICENSE                      # MIT
```

## Environment

* Windows / Linux / macOS with Python 3.10+ (developed on Python 3.13).
* Install dependencies:

```bash
pip install -r requirements.txt
```

Core packages: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`.
A LaTeX distribution with the `IEEEtran` class is needed
only to rebuild the manuscript PDF.

## Reproduction

All scripts resolve their data paths relative to their own location
(`__file__`); no path editing is required. Run them from the `analysis/`
directory (some scripts import sibling modules).

```bash
# Stage 1 — exact batching (seconds)
python Q1_精确求解.py

# Stage 2 — routing: candidate enumeration + greedy scheduling (~1-2 min, DEM)
python Q2_精确调度.py
python Q2_四指标非支配.py
python Q2_约束校验.py

# Stage 3 — MHPRS relay construction + independent verification (~minutes)
python Q3_TD-CJO_v2.py
python Q3_复核.py

# Problem-3 independent first-principles rebuild (see analysis/q3_rebuild/)
python q3_rebuild/01_轨迹与需中继点.py      # trajectory + relay-required points
python q3_rebuild/02_冻结方案独立核验.py    # frozen-plan point check + missing hover points
python q3_rebuild/04_单悬停基线.py          # single-hover baseline (51.0 / 62.7 / 82.1 %)
python q3_rebuild/05_门限敏感性.py          # threshold sensitivity (2311 / 806 / 0)
python q3_rebuild/06_规范集合核验.py        # coverage on the canonical point set
python q3_rebuild/07_能耗SOC_核验方案.py    # energy/SOC under eta_c = 0.72
python q3_rebuild/08_固定方案可行性核验.py  # read-only feasibility of the fixed plan
python q3_rebuild/09_下界_中继机数.py       # >= 2 relay-airframe lower bound
python q3_rebuild/10_只读验证器.py          # read-only feasibility audit; writes to outputs_B

# Stage 4 — resource partitioning
python Q4_分区核算_v3.py
python Q4_TOPSIS.py

# Clustering benchmark (journal version)
cd baselines
python q4_cluster_baseline.py   # Table XI: K-means / Ward vs SC-MAE-RG

# Figures
cd ../../figures && python make_figs.py
```

### Reproduced headline results

| Result | Value |
|---|---|
| P1 exact batching | 18 trips, 59.24 kWh (0.69 % below FFD/BFD) |
| P2 recommended plan | 20 trips, 65.25 kWh, makespan 2.72 h, 31/31 hard deadlines met |
| P3 MHPRS coverage | all 11 713 sampled relay points (11 645 unique after removing 68 boundary duplicates) on the 1 s grid, 4 sorties, 2 relay UAVs |
| P3 MHPRS relay energy | 5.432 kWh rebuilt vs 5.433 kWh frozen (-0.02 %); 6 distinct hover points serving 7 service windows |
| P3 relay-concurrency lower bound | >= 2 relay airframes (Proposition 2) |
| P3 threshold sensitivity (-90 / -95 / -98 dBm) | relay-required points 2311 / 806 / 0; directly served areas 3 / 10 / 15 |
| P3 weakest relay link | -89.62 dBm (TR<->relay), 0.38 dB above the -90 dBm threshold |
| Joint transport + relay energy | 70.68 kWh |
| P4 recommended K = 2 partition | D = 31, G = 5, J = 0.613 (best partition among evaluated candidates satisfying the >= 3-area grouping rule) |

### Problem-3 independent verification

`analysis/q3_rebuild/` rebuilds every Problem-3 claim from the raw DEM and the
frozen plan alone: per-second trajectories, the relay-required point set, the
point-by-point coverage verdicts, the energy/SOC ledger and the relay-concurrency
lower bound. The canonical set is **11 713 relay-required points on the 1 s grid
(11 645 unique after removing 68 boundary duplicates)**; a 30-point residual
(0.26 %) separates it from the earlier 11 683-point log, all on line-of-sight
margins within ~1 dB of the -90 dBm threshold. The transfer-climb efficiency is
the **prior model value eta_c = 0.72** (overridable via `ETA_CLIMB`) — no parameter
is fitted to the frozen results — under which the rebuilt relay energy is
5.432 kWh against the frozen 5.433 kWh (-0.02 %). The two hover points missing
from the frozen log are reconstructed once in `02_冻结方案独立核验.py` under the
paper's Sec. 5.3.2 rule (full coverage first, then nearest to the segment
centroid); `06`/`07`/`08` all read that single hover-point chain, so coverage,
energy and feasibility are verified on identical coordinates. The single-hover
baseline (`04`) applies the **same** per-sortie energy model, return-SOC floor
(0.20) and two-stage charging rule (`T_chg = 500 + 1300*(1-SOC)` s) as the main
method; after unification the covered-point counts (5968 / 7347 / 9611 for
N = 1/2/3) are unchanged.

`10_只读验证器.py` audits the frozen plan without searching coordinates: it
rebuilds coverage, the arrival chain, the transfer clearance and the
machine/component occupancy from the frozen artifacts alone. It also quantifies
the clearance convention. The demand side is generated with a 6 m terrain
clearance and bilinear interpolation, whereas the coverage test uses the raw
DEM profile and nearest pixels; under the 6 m convention 9679 of 11713 points
pass (82.63 %), one reconstructed hover point falls 0.96 dB short on its
depot link, and relocating that point by 256 m restores 100 % coverage with
582 s of arrival slack, return SOC 0.41 and sortie energy 1.2 % above the
frozen value. The paper therefore reports the 100 % figure as tied to the
reconstruction convention and states the 6 m re-check explicitly.

Key deterministic settings:

* Random seed `20251015` for the P1/P2 random-instance families and for
  K-means.
* Typical wall time on an 8-core desktop CPU: P1 < 1 s; Q4 clustering < 1 s;
  the P2 structure enumeration and P3 per-second verification are the longest
  stages (single-digit minutes) and are embarrassingly parallel.

## Data provenance

The instance (dispatch center, 15 service areas, three transport UAV types,
relay UAV specifications, deadlines and the 30 m DEM of Zhenlong Township,
Guangxi) was provided by the **China Postgraduate Mathematical Contest in
Modeling** for the 2026 contest. It is included here for research
reproducibility. Rights to the underlying geographic and contest tables
remain with their originators; the code in this repository is released under
the MIT License.

The DEM is shipped as the MATLAB `.mat` array actually consumed by the
scripts (`analysis/数据/.../镇龙乡及周边30米DEM.mat`, ≈6.4 MB).

## Citation

A BibTeX entry will be provided once the manuscript is accepted. For now
please cite the title and authors listed above.
