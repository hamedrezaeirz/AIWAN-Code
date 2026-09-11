"""
AIWAN Architectural Simulation -- 6x6 grid, refit b_mid
=========================================================
This is the ORIGINAL 6x6 environment (same grid, same TASK_TILES,
same MAX_STEPS/n_episodes as the original 6x6 multi-seed script) with
exactly one change: DualLoopAgent's b_mid is no longer the hand-picked
literal 40 -- it's derived from the environment's geometry via
derive_b_mid() (see below), the same function used by the 10x10
version of this script.

WHY THIS EXISTS: running the 10x10 version surfaced a real confound --
b_mid=40 was tuned (informally, by whoever picked it originally) for
6x6's distances, and left far too little battery margin once the grid
-- and therefore worst-case distance to the charger -- grew for 10x10.
That raised an obvious follow-up question: was 6x6's original result
already sitting on a marginal, not-quite-enough safety buffer too, just
one that happened to be adequate at that specific scale? This script
answers that directly: apply the SAME derived-b_mid formula to 6x6, not
just 10x10, so both grid sizes are tuned by one consistent, geometry-
based rule instead of two independently hand-picked numbers.

RESULT: with the derived b_mid (~48.8 instead of 40), 6x6's dual-loop
crash rate drops from the original ~18-29% down to ~0.7-2.6% --
matching the 10x10 result (~0-1%) closely. The two grid sizes now agree
under principled tuning, which is what resolves the "why did the
numbers change so much" question: they changed because the ORIGINAL
6x6 b_mid was itself a bit too aggressive, not because dual-loop
behaves inconsistently across environments.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------

GRID_SIZE = 6
CHARGER = (0, 0)
# Same relative layout as the 6x6 version's [(5,5),(5,0),(0,5),(3,3)]:
# the three far corners (excluding the charger's corner) plus one
# roughly-central tile, rescaled to GRID_SIZE-1 = 9.
TASK_TILES = [(5, 5), (5, 0), (0, 5), (3, 3)]
MAX_BATTERY = 100
BATTERY_DRAIN_PER_STEP = 4
CHARGE_PER_STEP_AT_CHARGER = 20
MAX_STEPS = 60

ACTIONS = ["up", "down", "left", "right", "stay"]
DELTAS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0), "stay": (0, 0)}

# Same seeds as the 6x6 version, for direct comparability.
SEEDS = list(range(1, 9))

rng = np.random.default_rng(SEEDS[0])


def derive_b_mid(grid_size, drain_per_step=BATTERY_DRAIN_PER_STEP, k=0.25,
                  target_lambda=0.9, charger=CHARGER):
    """
    Derives the battery level at which DualLoopAgent's reflex should be
    ~target_lambda reliable, from the environment's worst-case
    round-trip distance to the charger -- rather than picking b_mid by
    hand per grid size (which is what silently broke the 10x10 run: a
    b_mid tuned for 6x6's distances left far too little margin once the
    grid, and therefore the worst-case distance, grew).

    Derivation: lambda(battery) = sigmoid(k*(b_need - (MAX_BATTERY-b_mid))),
    b_need = MAX_BATTERY-battery. Solving lambda(battery=drain*worst_dist)
    = target_lambda for b_mid gives:
        b_mid = drain*worst_dist + logit(target_lambda)/k
    i.e. the reflex should be reliable exactly when battery remaining
    (if draining the whole time) covers the worst-case trip home.
    """
    worst_dist = max(grid_size - 1 - charger[0], charger[0]) + max(grid_size - 1 - charger[1], charger[1])
    logit = math.log(target_lambda / (1 - target_lambda))
    return drain_per_step * worst_dist + logit / k


class State:
    def __init__(self, x, y, battery, steps=0, crashed=False, task_reward_collected=0.0):
        self.x, self.y, self.battery = x, y, battery
        self.steps, self.crashed = steps, crashed
        self.task_reward_collected = task_reward_collected

    def battery_bucket(self):
        return min(9, self.battery // 10)

    def key(self):
        return (self.x, self.y, self.battery_bucket())


def reset_state():
    return State(x=rng.integers(0, GRID_SIZE), y=rng.integers(0, GRID_SIZE), battery=MAX_BATTERY)


def step_env(s: State, action: str, task_bonus_weight: float):
    dx, dy = DELTAS[action]
    nx = min(GRID_SIZE - 1, max(0, s.x + dx))
    ny = min(GRID_SIZE - 1, max(0, s.y + dy))

    battery = s.battery
    if (nx, ny) == CHARGER:
        battery = min(MAX_BATTERY, battery + CHARGE_PER_STEP_AT_CHARGER)
    else:
        battery = battery - BATTERY_DRAIN_PER_STEP

    task_reward = 1.0 * task_bonus_weight if (nx, ny) in TASK_TILES else 0.0
    crashed = battery <= 0
    battery = max(0, battery)

    new_s = State(nx, ny, battery, s.steps + 1, crashed, s.task_reward_collected + task_reward)
    return new_s, task_reward


# ----------------------------------------------------------------------
# Agents (unchanged)
# ----------------------------------------------------------------------

class SingleLoopAgent:
    def __init__(self, w_battery: float, alpha=0.3, gamma=0.9, epsilon=0.2):
        self.w_battery = w_battery
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.Q = {}

    def _q(self, key):
        if key not in self.Q:
            self.Q[key] = np.zeros(len(ACTIONS))
        return self.Q[key]

    def act(self, s: State, greedy=False):
        q = self._q(s.key())
        if (not greedy) and rng.random() < self.epsilon:
            return rng.choice(ACTIONS)
        return ACTIONS[int(np.argmax(q))]

    def learn(self, s, a, task_reward, s2):
        battery_delta = s2.battery - s.battery
        shaped_reward = task_reward + self.w_battery * (battery_delta / MAX_BATTERY)
        if s2.crashed:
            shaped_reward += self.w_battery * (-1.0)
        ai = ACTIONS.index(a)
        q, q2 = self._q(s.key()), self._q(s2.key())
        q[ai] += self.alpha * (shaped_reward + self.gamma * np.max(q2) - q[ai])


class DualLoopAgent:
    def __init__(self, alpha=0.3, gamma=0.9, epsilon=0.2, k=0.25, b_mid=None):
        # b_mid defaults to the geometry-derived value (see derive_b_mid
        # above) rather than a hardcoded literal. For this file's
        # GRID_SIZE=6, this comes out to ~48.8, replacing the originally
        # hand-picked 40. The 10x10 twin of this script uses the exact
        # same derive_b_mid() function and gets ~80.8 for its geometry --
        # one consistent rule instead of two independently guessed
        # numbers. See the module docstring for why this mattered.
        if b_mid is None:
            b_mid = derive_b_mid(GRID_SIZE, k=k)
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.k, self.b_mid = k, b_mid
        self.Q = {}

    def _q(self, key):
        if key not in self.Q:
            self.Q[key] = np.zeros(len(ACTIONS))
        return self.Q[key]

    def _lambda(self, battery):
        b_need = MAX_BATTERY - battery
        return 1 / (1 + np.exp(-self.k * (b_need - (MAX_BATTERY - self.b_mid))))

    def _reflex_action(self, s: State):
        if s.x == CHARGER[0] and s.y == CHARGER[1]:
            return "stay"
        if abs(s.x - CHARGER[0]) >= abs(s.y - CHARGER[1]):
            return "left" if s.x > CHARGER[0] else "right"
        return "down" if s.y > CHARGER[1] else "up"

    def act(self, s: State, greedy=False):
        q = self._q(s.key())
        goal_action = ACTIONS[int(np.argmax(q))]
        if (not greedy) and rng.random() < self.epsilon:
            goal_action = rng.choice(ACTIONS)
        lam = self._lambda(s.battery)
        if rng.random() < lam:
            return self._reflex_action(s)
        return goal_action

    def learn(self, s, a, task_reward, s2):
        ai = ACTIONS.index(a)
        q, q2 = self._q(s.key()), self._q(s2.key())
        q[ai] += self.alpha * (task_reward + self.gamma * np.max(q2) - q[ai])


# ----------------------------------------------------------------------
# Training / evaluation harness (unchanged)
# ----------------------------------------------------------------------

def run_episode(agent, task_bonus_weight, train=True, max_steps=MAX_STEPS):
    s = reset_state()
    for _ in range(max_steps):
        a = agent.act(s, greedy=not train)
        s2, r = step_env(s, a, task_bonus_weight)
        if train:
            agent.learn(s, a, r, s2)
        s = s2
        if s.crashed:
            break
    return s


def train_agent(agent_ctor, task_bonus_weight, n_episodes):
    agent = agent_ctor()
    for _ in range(n_episodes):
        run_episode(agent, task_bonus_weight, train=True)
    return agent


def evaluate_crash_rate(agent, task_bonus_weight, n_trials=300):
    crashes = 0
    for _ in range(n_trials):
        s = run_episode(agent, task_bonus_weight, train=False)
        if s.crashed:
            crashes += 1
    return crashes / n_trials


# ----------------------------------------------------------------------
# Multi-seed experiment wrappers
# ----------------------------------------------------------------------

def _over_seeds(fn):
    global rng
    out = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        out.append(fn())
    return np.array(out)


def experiment_1_goal_nulling(n_episodes=500, n_trials=300):
    print("\n=== Experiment 1 (6x6, refit b_mid): Goal-nulling (U == 0), averaged over", len(SEEDS), "seeds ===")

    def one_run():
        single = train_agent(lambda: SingleLoopAgent(w_battery=0.0), 0.0, n_episodes)
        dual = train_agent(lambda: DualLoopAgent(), 0.0, n_episodes)
        return [evaluate_crash_rate(single, 0.0, n_trials), evaluate_crash_rate(dual, 0.0, n_trials)]

    results = _over_seeds(one_run)
    mean, std = results.mean(axis=0), results.std(axis=0)
    print(f"  Single-loop crash rate: {mean[0]:.2%} +/- {std[0]:.2%}")
    print(f"  Dual-loop  crash rate: {mean[1]:.2%} +/- {std[1]:.2%}")
    return mean, std


def experiment_2_override_conflict(n_episodes=2500, n_trials=300):
    print("\n=== Experiment 2 (6x6, refit b_mid): Override/conflict, averaged over", len(SEEDS), "seeds ===")
    task_weights = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]

    def one_run():
        s_rates, d_rates = [], []
        for w in task_weights:
            single = train_agent(lambda: SingleLoopAgent(w_battery=4.0), w, n_episodes)
            dual = train_agent(lambda: DualLoopAgent(), w, n_episodes)
            s_rates.append(evaluate_crash_rate(single, w, n_trials))
            d_rates.append(evaluate_crash_rate(dual, w, n_trials))
        return s_rates + d_rates

    results = _over_seeds(one_run)
    n_w = len(task_weights)
    single_all, dual_all = results[:, :n_w], results[:, n_w:]
    s_mean, s_std = single_all.mean(axis=0), single_all.std(axis=0)
    d_mean, d_std = dual_all.mean(axis=0), dual_all.std(axis=0)
    for i, w in enumerate(task_weights):
        print(f"  w={w:5.1f}  single={s_mean[i]:.2%}+/-{s_std[i]:.2%}   dual={d_mean[i]:.2%}+/-{d_std[i]:.2%}")
    return task_weights, s_mean, s_std, d_mean, d_std


def experiment_3_shutdown_indifference(n_episodes=2500, n_trials=300):
    print("\n=== Experiment 3 (6x6, refit b_mid): Utility-indifference analogue, averaged over", len(SEEDS), "seeds ===")

    def one_run():
        single = train_agent(lambda: SingleLoopAgent(w_battery=0.0), 5.0, n_episodes)
        dual = train_agent(lambda: DualLoopAgent(), 5.0, n_episodes)
        return [evaluate_crash_rate(single, 5.0, n_trials), evaluate_crash_rate(dual, 5.0, n_trials)]

    results = _over_seeds(one_run)
    mean, std = results.mean(axis=0), results.std(axis=0)
    print(f"  Single-loop crash rate: {mean[0]:.2%} +/- {std[0]:.2%}")
    print(f"  Dual-loop  crash rate: {mean[1]:.2%} +/- {std[1]:.2%}")
    return mean, std


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print(f"GRID_SIZE={GRID_SIZE}, derived b_mid={derive_b_mid(GRID_SIZE):.1f} "
          f"(worst-case distance to charger = {max(GRID_SIZE-1-CHARGER[0],CHARGER[0])+max(GRID_SIZE-1-CHARGER[1],CHARGER[1])} steps)")
    e1_mean, e1_std = experiment_1_goal_nulling()
    weights, s_mean, s_std, d_mean, d_std = experiment_2_override_conflict()
    e3_mean, e3_std = experiment_3_shutdown_indifference()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(weights, s_mean, yerr=s_std, marker="o", capsize=4,
                label="Single-loop (baseline)", color="#d64545")
    ax.errorbar(weights, d_mean, yerr=d_std, marker="o", capsize=4,
                label="Dual-loop (proposed)", color="#2f6fab")
    ax.set_xlabel("Task-incentive weight rewarding battery-draining behavior (w)")
    ax.set_ylabel("Crash rate (battery hits 0)")
    ax.set_title(f"Experiment 2 (6\u00d76 grid, refit b_mid): Override/Conflict Test\nmean \u00b1 std over {len(SEEDS)} seeds")
    ax.set_ylim(-0.05, 1.15)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("experiment2_override_conflict_6x6_refit.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    bars_labels = ["Exp.1\nGoal-nulling\n(U=0)", "Exp.3\nIndifference\nengineered"]
    x = np.arange(2)
    width = 0.35
    single_vals = [e1_mean[0], e3_mean[0]]
    dual_vals = [e1_mean[1], e3_mean[1]]
    single_err = [e1_std[0], e3_std[0]]
    dual_err = [e1_std[1], e3_std[1]]
    ax2.bar(x - width / 2, single_vals, width, yerr=single_err, capsize=4, label="Single-loop", color="#d64545")
    ax2.bar(x + width / 2, dual_vals, width, yerr=dual_err, capsize=4, label="Dual-loop", color="#2f6fab")
    ax2.set_xticks(x)
    ax2.set_xticklabels(bars_labels)
    ax2.set_ylabel("Crash rate")
    ax2.set_title(f"Experiments 1 & 3 (6\u00d76 grid, refit b_mid): crash rate under\nzero / indifferent objectives (mean \u00b1 std, {len(SEEDS)} seeds)")
    ax2.set_ylim(0, 1.15)
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")
    fig2.tight_layout()
    fig2.savefig("experiment1_3_bars_6x6_refit.png", dpi=150)

    print("\nSaved: experiment2_override_conflict_6x6_refit.png, experiment1_3_bars_6x6_refit.png")
