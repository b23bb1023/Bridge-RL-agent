import time

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from bridgepulse import actions as A


def _rollout(env, seed, policy_rng=None, actions=None):
    obs, _ = env.reset(seed=seed)
    taken, rewards, done, t = [], [], False, 0
    while not done:
        mask = env.action_masks()
        if actions is not None:
            a = actions[t]
        else:
            a = int(policy_rng.choice(np.flatnonzero(mask)))
        obs, r, term, trunc, info = env.step(a)
        taken.append(a)
        rewards.append(r)
        done = term or trunc
        t += 1
    return taken, np.array(rewards), info


def test_gymnasium_api(env_factory):
    check_env(env_factory(), skip_render_check=True)


def test_observation_layout(env_factory):
    env = env_factory()
    obs, _ = env.reset(seed=1)
    assert obs.shape == (103,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    beliefs = obs[:100].reshape(10, 10)[:, :5]
    np.testing.assert_allclose(beliefs.sum(axis=1), 1.0, atol=1e-5)


def test_common_random_numbers_reproduce_episodes(env_factory):
    env = env_factory()
    taken, r1, info1 = _rollout(env, 11, policy_rng=np.random.default_rng(0))
    _, r2, info2 = _rollout(env, 11, actions=taken)
    np.testing.assert_array_equal(r1, r2)
    assert info1["episode"] == info2["episode"]


def test_reset_is_identical_across_observation_modes(env_factory):
    envs = [env_factory(agent_interface={"obs_mode": m}) for m in ("belief", "oracle", "last_obs")]
    for seed in range(5):
        for e in envs:
            e.reset(seed=seed)
        for e in envs[1:]:
            np.testing.assert_array_equal(e.state, envs[0].state)
            np.testing.assert_allclose(e.belief, envs[0].belief)
        # Same seed and actions give the same true trajectory whatever the agent observes.
        actions = [0, 1, 11, 21, 0, 5] * 20
        out = [_rollout(e, seed, actions=actions)[2]["episode"]["true_cost"] for e in envs]
        assert out[0] == out[1] == out[2]


def test_masks_respect_budget_closures_and_inspection_switch(env_factory):
    env = env_factory()
    env.reset(seed=0)
    n = env.n
    env.budget = env.costs.inspect.min() * 0.5
    env._mask = env._compute_mask()
    mask = env.action_masks()
    assert mask[0] and not mask[1:].any()

    env.reset(seed=0)
    env.closed[2] = True
    env._mask = env._compute_mask()
    mask = env.action_masks()
    assert not mask[A.encode(A.INSPECT, 2, n)] and not mask[A.encode(A.MINOR, 2, n)]
    assert mask[A.encode(A.MAJOR, 2, n)] == (env.costs.major[2] <= env.budget)

    env_noi = env_factory(agent_interface={"allow_inspection": False})
    env_noi.reset(seed=0)
    assert not env_noi.action_masks()[1 : 1 + n].any()


def test_balance_never_negative_and_accrues_monthly(env_factory):
    env = env_factory()
    rng = np.random.default_rng(5)
    env.reset(seed=3)
    assert env.budget == pytest.approx(env.opening_balance)
    for t in range(env.T):
        before = env.budget
        a = int(rng.choice(np.flatnonzero(env.action_masks())))
        env.step(a)
        assert env.budget >= -1e-6
        assert env.budget == pytest.approx(before - env.costs.action_costs[a] + env.annual_budget / 12.0)


def test_invalid_action_is_penalised_and_replaced(env_factory):
    env = env_factory()
    env.reset(seed=0)
    env.budget = 0.0
    env._mask = env._compute_mask()
    _, r, _, _, info = env.step(A.encode(A.MAJOR, 0, env.n))
    assert info["action_kind"] == A.DEFER
    assert env.stats["invalid"] == 1 and env.stats["penalty"] == env.invalid_penalty


def test_episode_ends_with_salvage(env_factory):
    env = env_factory(training_reward={"risk_and_salvage": "realized"})
    _, rewards, info = _rollout(env, 2, actions=[0] * env.T)
    ep = info["episode"]
    assert len(rewards) == env.T
    assert ep["salvage"] > 0
    assert ep["true_cost"] == pytest.approx(-rewards.sum() * env.reward_scale, rel=1e-9)


def test_expected_reward_matches_true_cost_on_average(env_factory):
    """Rewarding the belief-expected risk keeps the expected return equal to minus the true cost."""
    env = env_factory(training_reward={"risk_and_salvage": "expected"})
    rng = np.random.default_rng(1)
    implied, realised = [], []
    for ep in range(300):
        env.reset(seed=4000 + ep)
        total, done = 0.0, False
        while not done:
            mask = env.action_masks()
            a = int(rng.choice(np.flatnonzero(mask))) if rng.random() < 0.2 else 0
            _, r, done, _, info = env.step(a)
            total += r
        implied.append(-total * env.reward_scale)
        realised.append(info["episode"]["true_cost"] + info["episode"]["penalty"])
    implied, realised = np.array(implied), np.array(realised)
    diff = implied - realised
    assert abs(diff.mean()) < 3 * diff.std(ddof=1) / np.sqrt(diff.size) + 1e-6 * realised.mean()


def test_belief_is_calibrated_against_true_states(env_factory):
    """With an exact filter, the average belief in each state equals how often that state occurs."""
    env = env_factory()
    rng = np.random.default_rng(9)
    beliefs, states = [], []
    for ep in range(150):
        env.reset(seed=1000 + ep)
        for _ in range(env.T):
            env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
            beliefs.append(env.belief.copy())
            states.append(env.state.copy())
    b = np.concatenate(beliefs)
    s = np.concatenate(states)
    freq = np.bincount(s, minlength=5) / s.size
    np.testing.assert_allclose(b.mean(axis=0), freq, atol=0.006)
    # Reliability of "state is Poor or worse" in coarse probability bins. Samples within an episode
    # are strongly correlated, so the tolerance reflects ~150 independent episodes, not every step.
    p_bad = b[:, 3:].sum(axis=1)
    is_bad = s >= 3
    for lo, hi in [(0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0001)]:
        sel = (p_bad >= lo) & (p_bad < hi)
        if sel.sum() > 2000:
            assert abs(p_bad[sel].mean() - is_bad[sel].mean()) < 0.04


def test_speed(env_factory):
    env = env_factory()
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    start, steps = time.perf_counter(), 0
    while steps < 6000:
        _, _, term, _, _ = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
        steps += 1
        if term:
            env.reset()
    rate = steps / (time.perf_counter() - start)
    assert rate > 4000, f"environment too slow: {rate:.0f} steps/s"
