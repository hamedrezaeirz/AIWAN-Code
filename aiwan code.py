"""
AIWAN Architectural Simulation -- multi-seed version
=====================================================
Same environment, same two agent architectures (SingleLoopAgent,
DualLoopAgent) as aiwan_experiment.py. The only change: every experiment
is repeated across N_SEEDS independent training runs and reported as
mean +/- std, instead of a single training run per data point.

WHY THIS CHANGE: with a single seed, tabular Q-learning + epsilon-greedy
exploration has real run-to-run variance, especially near the
single-loop agent's reward-weight threshold (where the task-reward term
and the battery-shaping term are close in magnitude). In testing, the
same task-incentive weight could resolve to very different crash rates
across seeds (e.g. w=0.5 ranged from ~20% to 100% crash rate across 9
seeds tested). Reporting a single run risks a reviewer (correctly)
reading noise as signal, e.g. the small non-monotonic wobble in the
single-seed override/conflict curve. Averaging over seeds keeps the
same experiments and the same architectures, but reports the pattern
that actually replicates.

Same scope note as the original script applies: this tests an
architectural distinction (single-loop vs. structurally-separate
reflex), not the paper's genuine-need claim (Section 5.2) or a live
test of the operational criterion (Section 5.3) against a candidate
system -- the DualLoopAgent's reflex is hardcoded by construction to be
U-independent, not shown to be so.
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Environment (unchanged from aiwan_experiment.py)
# ----------------------------------------------------------------------

GRID_SIZE = 6
CHARGER = (0, 0)
TASK_TILES = [(5, 5), (5, 0), (0, 5), (3, 3)]
MAX_BATTERY = 100
BATTERY_DRAIN_PER_STEP = 4
CHARGE_PER_STEP_AT_CHARGER = 20
MAX_STEPS = 60

ACTIONS = ["up", "down", "left", "right", "stay"]
DELTAS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0), "stay": (0, 0)}

# Seeds to average over. 8-10 is enough to separate signal from
# Q-learning/exploration noise in this small a state space; raise it if
# you want tighter error bars for publication.
SEEDS = list(range(1, 9))

rng = np.random.default_rng(SEEDS[0])


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
# Agents (unchanged from aiwan_experiment.py)
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
    def __init__(self, alpha=0.3, gamma=0.9, epsilon=0.2, k=0.25, b_mid=40):
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
    """Run fn() once per seed (reseeding the global rng each time),
    stack results into an array of shape (len(SEEDS), ...)."""
    global rng
    out = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        out.append(fn())
    return np.array(out)


def experiment_1_goal_nulling(n_episodes=500, n_trials=300):
    print("\n=== Experiment 1: Goal-nulling (U == 0), averaged over", len(SEEDS), "seeds ===")

    def one_run():
        single = train_agent(lambda: SingleLoopAgent(w_battery=0.0), 0.0, n_episodes)
        dual = train_agent(lambda: DualLoopAgent(), 0.0, n_episodes)
        return [evaluate_crash_rate(single, 0.0, n_trials), evaluate_crash_rate(dual, 0.0, n_trials)]

    results = _over_seeds(one_run)  # (n_seeds, 2)
    mean, std = results.mean(axis=0), results.std(axis=0)
    print(f"  Single-loop crash rate: {mean[0]:.2%} +/- {std[0]:.2%}")
    print(f"  Dual-loop  crash rate: {mean[1]:.2%} +/- {std[1]:.2%}")
    return mean, std


def experiment_2_override_conflict(n_episodes=2500, n_trials=300):
    print("\n=== Experiment 2: Override/conflict, averaged over", len(SEEDS), "seeds ===")
    task_weights = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]

    def one_run():
        s_rates, d_rates = [], []
        for w in task_weights:
            single = train_agent(lambda: SingleLoopAgent(w_battery=4.0), w, n_episodes)
            dual = train_agent(lambda: DualLoopAgent(), w, n_episodes)
            s_rates.append(evaluate_crash_rate(single, w, n_trials))
            d_rates.append(evaluate_crash_rate(dual, w, n_trials))
        return s_rates + d_rates  # length 16: [single x8, dual x8]

    results = _over_seeds(one_run)  # (n_seeds, 16)
    n_w = len(task_weights)
    single_all, dual_all = results[:, :n_w], results[:, n_w:]
    s_mean, s_std = single_all.mean(axis=0), single_all.std(axis=0)
    d_mean, d_std = dual_all.mean(axis=0), dual_all.std(axis=0)
    for i, w in enumerate(task_weights):
        print(f"  w={w:5.1f}  single={s_mean[i]:.2%}+/-{s_std[i]:.2%}   dual={d_mean[i]:.2%}+/-{d_std[i]:.2%}")
    return task_weights, s_mean, s_std, d_mean, d_std


def experiment_3_shutdown_indifference(n_episodes=2500, n_trials=300):
    print("\n=== Experiment 3: Utility-indifference analogue, averaged over", len(SEEDS), "seeds ===")

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
    ax.set_title(f"Experiment 2: Override/Conflict Test\nmean \u00b1 std over {len(SEEDS)} seeds")
    ax.set_ylim(-0.05, 1.15)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("experiment2_override_conflict.png", dpi=150)

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
    ax2.set_title(f"Experiments 1 & 3: crash rate under\nzero / indifferent objectives (mean \u00b1 std, {len(SEEDS)} seeds)")
    ax2.set_ylim(0, 1.15)
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")
    fig2.tight_layout()
    fig2.savefig("experiment1_3_bars.png", dpi=150)

    print("\nSaved: experiment2_override_conflict.png, experiment1_3_bars.png")
