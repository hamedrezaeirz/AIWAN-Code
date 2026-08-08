# AIWAN-Code

Reference implementation and experiment scripts for **AIWAN: AI With an Artificial Need** (Rezaei, 2026c), a companion model to **AIWBN: AI With a Biological Need** (Rezaei, 2026b).

This repository contains the multi-seed simulation used to produce the results reported in AIWAN §5–§6 (Table 1, Figures 1–2): a grid-world battery-management task comparing a single-loop (reward-shaped) agent against a dual-loop agent with a structurally-separate, U-independent regulatory reflex.

## What this code tests — and what it doesn't

This script tests an **architectural** claim: whether a structurally-separate regulatory loop produces different behavior than reward-shaping alone, at fixed, modest optimization capacity. It does **not**:

- Test AIWBN's genuine-need claim (need grounded in living tissue) — the need here (`battery`) is an ordinary represented quantity by design (AIWAN §3.3, §4.1).
- Provide a live test of AIWBN's operational criterion (§5.3) against a candidate system.
- Demonstrate or rule out the fold-back/collapse prediction of AIWAN §4.2 (that the dual-loop advantage should erode as the goal process's optimization capacity increases). The dual-loop agent's reflex, `_reflex_action`, is hardcoded, not learned — it is U-independent by construction, not because the system discovered independence. See AIWAN §7.3 for the experiment that would test the collapse prediction directly.

## Requirements

See `requirements.txt`. Python 3.9+.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python aiwan code.py
```

This runs all three experiments (Goal-nulling, Override/conflict, Utility-indifference analogue), each averaged over 8 seeds (`SEEDS = range(1, 9)`), and saves two figures:

- `experiment2_override_conflict.png` — Figure 1 in the paper
- `experiment1_3_bars.png` — Figure 2 in the paper

Console output reports mean ± std crash rate for each condition, matching AIWAN Table 1.

**Runtime warning:** a full run (8 seeds × up to 2,500 training episodes per data point × 8 task-incentive weights in Experiment 2) takes on the order of tens of minutes on a single core. There is currently no `--quick` flag; to sanity-check the code faster, reduce `SEEDS` and the `n_episodes` arguments to `experiment_1_goal_nulling`, `experiment_2_override_conflict`, and `experiment_3_shutdown_indifference` directly in the script. Numbers produced this way are for sanity-checking only and should not be cited — the paper's reported results use the full-scale, unmodified parameters above.

## Results reference (AIWAN Table 1)

| Experiment | Single-loop crash rate | Dual-loop crash rate |
|---|---|---|
| 1 — Goal-nulling (U ≡ 0) | 100.00% ± 0.00% | 17.83% ± 2.22% |
| 2 — Override (w=8) | 100.00% ± 0.00% | 24.54% ± 9.53% |
| 3 — Indifference engineered | 100.00% ± 0.00% | 27.29% ± 7.49% |

## Environment

- 6×6 grid-world, single charging tile at (0,0), four task tiles.
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
