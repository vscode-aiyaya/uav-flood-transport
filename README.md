# Coupled UAV Transportation–Communication Scheduling in Mountain Flood Disaster Relief

Reproducibility repository accompanying the manuscript:

> **A Unified Multi-Model Framework for Coupled UAV Transportation and
> Communication Scheduling in Mountain Flood Disaster Relief**
> Yang Xin, Zhao Zidong, Song Yuwei.
> *Manuscript, 2026* (preprint/journal reference to be added).

The repository contains the full four-stage model chain — exact energy-aware
bin packing (Problem 1), heterogeneous routing with battery turnover
(Problem 2), multi-hover-point relay scheduling / MHPRS (Problem 3), and
partition-based resource accounting / SC-MAE-RG (Problem 4) — together with
the external solver/clustering benchmarks added in the journal version.

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
│   ├── Q4_分区核算_v3.py        # P4 SC-MAE-RG resource accounting
│   ├── Q4_TOPSIS.py / Q4_绘图_v3.py
│   ├── *.csv / *.json           # frozen plans, logs and audit artifacts
│   └── baselines/               # external benchmarks (journal version)
│       ├── q2_cpsat_baseline.py # OR-Tools CP-SAT vs greedy, real instance
│       ├── q2_cpsat_family.py   # 30-instance robustness family + Wilcoxon test
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

Core packages: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`,
`ortools` (CP-SAT). A LaTeX distribution with the `IEEEtran` class is needed
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

# Stage 4 — resource partitioning
python Q4_分区核算_v3.py
python Q4_TOPSIS.py

# External benchmarks (journal version)
cd baselines
python q2_cpsat_baseline.py     # Table VI: CP-SAT vs greedy, <1 s solve
python q2_cpsat_family.py       # 30 synthetic instances, <=30 s per instance
python q4_cluster_baseline.py   # Table XI: K-means / Ward vs SC-MAE-RG

# Figures
cd ../../figures && python make_figs.py
```

### Reproduced headline results

| Result | Value |
|---|---|
| P1 exact batching | 18 trips, 59.24 kWh (0.69 % below FFD/BFD) |
| P2 recommended plan | 20 trips, 65.25 kWh, makespan 2.72 h, 31/31 hard deadlines met |
| CP-SAT vs greedy (fixed 20 trips) | 2.11 h vs 2.46 h makespan; 64.31 vs 67.38 kWh; proven optimal in 0.53 s |
| CP-SAT family (30 instances) | 2.07±0.28 vs 2.47±0.33 h, win/tie/loss 30/0/0, Wilcoxon p < 1e-8 |
| P3 MHPRS coverage | 11 683 / 11 683 relay-required points (100 %), 4 sorties, 2 relay UAVs |
| Joint transport + relay energy | 70.68 kWh |
| P4 recommended K = 2 partition | D = 31, G = 5, J = 0.613 (best valid partition vs all clustering baselines) |

Key deterministic settings:

* Random seed `20251015` for the P1/P2 random-instance families and for
  K-means; CP-SAT uses 8 workers and a 30 s (family) / 300 s (real instance
  cap) time limit.
* All CP-SAT durations are rounded **up** to whole seconds, so every
  CP-feasible schedule is strictly feasible under the continuous physical
  model; `q2_cpsat_baseline.py` additionally re-verifies the returned
  schedule against the continuous energy/time model in Python
  (`independent_verify.passed = true` in `q2_cpsat_real.json`).
* Typical wall time on an 8-core desktop CPU: P1 < 1 s; P2 scheduling
  benchmark 0.53 s (proven optimal); Q4 clustering < 1 s; the P2 structure
  enumeration and P3 per-second verification are the longest stages
  (single-digit minutes) and are embarrassingly parallel.

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
