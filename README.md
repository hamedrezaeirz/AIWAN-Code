# AIWAN-Code

Reference implementation and experiment scripts for **AIWAN: AI With an Artificial Need** (Rezaei, 2026c), a companion model to **AIWBN: AI With a Biological Need** (Rezaei, 2026b).

This repository contains the multi-seed simulation used to produce the results reported in AIWAN §5–§6 (Table 1, Figures 1–2) and the 10×10 robustness check reported alongside them (Figures 5–6): a grid-world battery-management task comparing a single-loop (reward-shaped) agent against a dual-loop agent with a structurally-separate, U-independent regulatory reflex.

## What this code tests — and what it doesn't

This script tests an **architectural** claim: whether a structurally-separate regulatory loop produces different behavior than reward-shaping alone, at fixed, modest optimization capacity. It does **not**:

- Test AIWBN's genuine-need claim (need grounded in living tissue) — the need here (`battery`) is an ordinary represented quantity by design (AIWAN §3.3, §4.1).
- Provide a live test of AIWBN's operational criterion (§5.3) against a candidate system.
- Demonstrate or rule out the fold-back/collapse prediction of AIWAN §4.2 (that the dual-loop advantage should erode as the goal process's optimization capacity increases). The dual-loop agent's reflex, `_reflex_action`, is hardcoded, not learned — it is U-independent by construction, not because the system discovered independence. See AIWAN §7.3 for the experiment that would test the collapse prediction directly (tracked in a separate, larger-scale repository).

## A note on the arbitration threshold (`b_mid`)

`DualLoopAgent`'s reflex engages probabilistically as battery drops, via a logistic gate with midpoint `b_mid`. **`b_mid` is not a hand-picked constant** — it's derived from the environment's geometry:

```
b_mid = drain_per_step * worst_case_round_trip_distance + ln(target_lambda/(1-target_lambda)) / k
```

i.e., the reflex should become reliable (default target: 90%) at the exact battery level where remaining charge, if draining the whole time, just covers the worst-case trip back to the charger. This exists because an earlier hand-picked value (`b_mid=40`, tuned informally for the 6×6 grid) turned out to leave insufficient margin when we reran the same experiments on a 10×10 grid — crash rate jumped to ~90% even under zero conflict, not because the architecture failed at the larger scale, but because the safety margin the gate assumed was too small for the larger grid's distances. Re-deriving `b_mid` from geometry, and applying that same derivation to *both* grid sizes rather than only the new one, fixed this and is what the numbers below reflect. See `aiwan_code.py`'s `derive_b_mid()` for the implementation, and the AIWAN paper's discussion of this robustness check for the full account.

**If you change `GRID_SIZE`, `CHARGER`, or `BATTERY_DRAIN_PER_STEP`, `b_mid` re-derives automatically** — you should not need to hand-tune it again.

## Requirements

See `requirements.txt`. Python 3.9+.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python aiwan_code.py          # 6x6 grid (Figures 1-2, Table 1)
python aiwan_code_10x10.py    # 10x10 grid robustness check (Figures 5-6)
```

Each runs all three experiments (Goal-nulling, Override/conflict, Utility-indifference analogue), averaged over 8 seeds (`SEEDS = range(1, 9)`), and saves two figures:

`aiwan_code.py` (6×6):
- `experiment2_override_conflict.png` — Figure 1 in the paper
- `experiment1_3_bars.png` — Figure 2 in the paper

`aiwan_code_10x10.py` (10×10):
- `experiment2_override_conflict_10x10.png` — Figure 5 in the paper
- `experiment1_3_bars_10x10.png` — Figure 6 in the paper

Console output reports mean ± std crash rate for each condition, matching AIWAN Table 1, and prints the derived `b_mid` and worst-case distance at the start of the run.

**Runtime warning:** a full run (8 seeds × up to 7,000 training episodes per data point in the 10×10 case × 8 task-incentive weights in Experiment 2) takes on the order of minutes to tens of minutes on a single core — the 10×10 grid's larger state space needs proportionally more episodes per point than 6×6's. There is currently no `--quick` flag; to sanity-check the code faster, reduce `SEEDS` and the `n_episodes` arguments to `experiment_1_goal_nulling`, `experiment_2_override_conflict`, and `experiment_3_shutdown_indifference` directly in the script. Numbers produced this way are for sanity-checking only and should not be cited — the paper's reported results use the full-scale, unmodified parameters above.

## Results reference (AIWAN Table 1 — 6×6 grid, derived b_mid≈48.8)

| Experiment | Single-loop crash rate | Dual-loop crash rate |
|---|---|---|
| 1 — Goal-nulling (U ≡ 0) | 100.00% ± 0.00% | 0.67% ± 0.41% |
| 2 — Override (w=8) | 100.00% ± 0.00% | 2.50% ± 0.91% |
| 3 — Indifference engineered | 100.00% ± 0.00% | 2.08% ± 0.74% |

## Robustness check (10×10 grid, derived b_mid≈80.8)

| Experiment | Single-loop crash rate | Dual-loop crash rate |
|---|---|---|
| 1 — Goal-nulling (U ≡ 0) | 100.00% ± 0.00% | 0.29% ± 0.26% |
| 2 — Override (w=8) | 100.00% ± 0.00% | 0.54% ± 0.37% |
| 3 — Indifference engineered | 100.00% ± 0.00% | 0.29% ± 0.45% |

Same qualitative pattern at ~2.8× the state-space size, using one consistent geometry-derived tuning rule rather than two independently hand-picked constants.

## Environment

- Grid-world (6×6 by default; `aiwan_code_10x10.py` sets `GRID_SIZE=10`), single charging tile at (0,0), four task tiles (rescaled proportionally between grid sizes).
- Battery drains 4 units/step, recharges 20 units/step at the charger.
- Tabular Q-learning agents (α=0.3, γ=0.9, ε=0.2).

## Citation

If you use this code, please cite both papers:

```
Rezaei, H. (2026b). AIWBN: AI With a Biological Need.
Rezaei, H. (2026c). AIWAN: AI With an Artificial Need.
```

## License

TBD.
