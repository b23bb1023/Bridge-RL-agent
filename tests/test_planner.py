import numpy as np

from baselines.heuristics import build
from baselines.rollout import RolloutPlanner, restore, snapshot


def _advance(env, months, seed=7):
    env.reset(seed=seed)
    rule = build("risk_greedy", {"min_benefit_per_dollar": 0.1})
    for _ in range(months):
        env.step(rule(env, None))
    return rule


def test_snapshot_restore_replays_identically(env_factory):
    env = env_factory()
    rule = _advance(env, 30)
    snap = snapshot(env)
    first = [env.step(rule(env, None))[1] for _ in range(20)]
    restore(env, snap)
    second = [env.step(rule(env, None))[1] for _ in range(20)]
    assert np.allclose(first, second)


def test_planner_leaves_the_real_episode_untouched(env_factory):
    env = env_factory()
    rule = _advance(env, 40)
    before = snapshot(env)
    RolloutPlanner(rule, n_rollouts=2, seed=0)(env, None)
    after = snapshot(env)
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, after[key]), key
        else:
            assert value == after[key], key


def test_planner_never_uses_the_true_condition(env_factory):
    env = env_factory()
    rule = _advance(env, 40)
    snap = snapshot(env)
    choices = []
    for flip in range(3):
        restore(env, snap)
        env.state = np.full(env.n, flip + 1)   # different hidden truth, same belief and history
        choices.append(RolloutPlanner(rule, n_rollouts=2, seed=0)(env, None))
    assert len(set(choices)) == 1
