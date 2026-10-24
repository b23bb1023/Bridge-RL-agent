import numpy as np
import pytest
import torch

from agents.dqn import DQNAgent, DQNConfig
from agents.networks import DecomposedQNetwork, SharedBridgeQNetwork
from agents.replay import NStepCollector, PrioritizedReplayBuffer, ReplayBuffer

OBS, ACTIONS, N = 103, 31, 10


def test_masked_agent_never_picks_infeasible_actions():
    agent = DQNAgent(OBS, ACTIONS, N, DQNConfig(arch="shared"), seed=0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        mask = rng.random(ACTIONS) < 0.3
        mask[0] = True
        obs = rng.random(OBS).astype(np.float32)
        for eps in (0.0, 1.0):
            assert mask[agent.act(obs, mask, eps)]


def test_shared_network_is_permutation_equivariant():
    torch.manual_seed(0)
    net = SharedBridgeQNetwork(N)
    obs = torch.rand(4, OBS)
    perm = torch.randperm(N)
    per = obs[:, : N * 10].reshape(4, N, 10)
    obs_perm = torch.cat([per[:, perm].reshape(4, -1), obs[:, N * 10 :]], dim=1)
    q, q_perm = net(obs), net(obs_perm)
    torch.testing.assert_close(q[:, 0], q_perm[:, 0], rtol=1e-5, atol=1e-5)
    for block in range(3):
        cols = 1 + block * N + torch.arange(N)
        torch.testing.assert_close(q[:, cols][:, perm], q_perm[:, cols], rtol=1e-5, atol=1e-5)


def test_decomposed_network_structure():
    torch.manual_seed(0)
    net = DecomposedQNetwork(N)
    obs = torch.rand(3, OBS)
    comp = net.components(obs)
    assert comp.shape == (3, N, ACTIONS)
    torch.testing.assert_close(net(obs), comp.sum(dim=1))
    # Work on bridge 2 changes bridge 5's component only through the price of the money spent.
    minor_2 = 1 + N + 2
    assert torch.all(comp[:, 5, minor_2] <= comp[:, 5, 0] + 1e-6)
    major_2 = 1 + 2 * N + 2
    assert torch.all(comp[:, 5, major_2] <= comp[:, 5, minor_2] + 1e-6)


def test_decomposed_agent_learns_from_vector_rewards():
    torch.set_num_threads(1)
    agent = DQNAgent(OBS, ACTIONS, N, DQNConfig(arch="decomposed", batch_size=16), seed=0)
    rng = np.random.default_rng(0)
    batch = {"obs": rng.random((16, OBS)).astype(np.float32), "action": rng.integers(0, ACTIONS, 16),
             "reward": -rng.random((16, N)).astype(np.float32), "next_obs": rng.random((16, OBS)).astype(np.float32),
             "discount": np.full(16, 0.97, np.float32), "next_mask": np.ones((16, ACTIONS), bool)}
    first = agent.update(batch)
    for _ in range(50):
        last = agent.update(batch)
    assert last < first


def test_reward_vector_adds_up_to_reward(env_factory):
    env = env_factory()
    env.reset(seed=8)
    rng = np.random.default_rng(2)
    for _ in range(env.T):
        _, r, done, _, info = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
        assert info["reward_vector"].shape == (env.n,)
        assert float(info["reward_vector"].sum()) == pytest.approx(r, rel=1e-5, abs=1e-6)


def test_checkpoint_round_trip(tmp_path):
    agent = DQNAgent(OBS, ACTIONS, N, DQNConfig(arch="flat"), seed=1)
    path = tmp_path / "agent.pt"
    agent.save(path, {"note": "x"})
    loaded = DQNAgent.load(path)
    obs = np.random.default_rng(2).random((5, OBS)).astype(np.float32)
    np.testing.assert_allclose(agent.q_values(obs), loaded.q_values(obs))
    assert loaded.meta["note"] == "x"


def test_prioritized_replay_samples_in_proportion_to_priority():
    buf = PrioritizedReplayBuffer(8, 1, 2, alpha=1.0, seed=0)
    for k in range(5):
        buf.add(np.array([k], np.float32), 0, 0.0, np.array([k], np.float32), 0.9, np.ones(2, bool))
    buf.update_priorities(np.arange(5), np.array([1.0, 1.0, 1.0, 1.0, 6.0]) - 1e-3)
    counts = np.zeros(5)
    for _ in range(2000):
        counts += np.bincount(buf.sample(10)["indices"], minlength=5)
    np.testing.assert_allclose(counts / counts.sum(), [0.1, 0.1, 0.1, 0.1, 0.6], atol=0.02)
    assert buf.sample(4)["weights"].max() == 1.0


def test_n_step_returns():
    buf = ReplayBuffer(16, 1, 2)
    col = NStepCollector(3, 0.5, buf)
    for t in range(4):
        col.push(np.array([t], np.float32), 0, 1.0, np.array([t + 1], np.float32), np.ones(2, bool), terminal=(t == 3))
    # Non-terminal 3-step return from t=0: 1 + 0.5 + 0.25, bootstrapped with 0.5^3.
    assert buf.reward[0] == pytest.approx(1.75) and buf.discount[0] == pytest.approx(0.125)
    # Episode end flushes the queue with no bootstrap.
    np.testing.assert_allclose(buf.reward[1:4], [1.75, 1.5, 1.0])
    np.testing.assert_allclose(buf.discount[1:4], 0.0)


def test_n_step_collector_keeps_each_observation_when_the_caller_reuses_its_array():
    # The training loop writes every new observation into the same array; queued transitions must not change.
    buf = ReplayBuffer(16, 2, 3, reward_dim=2)
    col = NStepCollector(3, 0.9, buf)
    obs = np.zeros((1, 2), np.float32)
    mask = np.ones(3, bool)
    for t in range(6):
        obs[0] = [t, 10 * t]
        reward = np.array([t, -t], np.float32)
        col.push(obs[0], t % 3, reward, np.array([t + 1, 10 * (t + 1)], np.float32), mask, terminal=(t == 5), mask=mask)
        reward[:] = 99.0   # the caller may also reuse its reward array
    for t in range(6):
        np.testing.assert_allclose(buf.obs[t], [t, 10 * t])
        assert buf.action[t] == t % 3
        steps = range(t, min(t + 3, 6))
        np.testing.assert_allclose(buf.reward[t], [sum(0.9 ** (k - t) * k for k in steps), -sum(0.9 ** (k - t) * k for k in steps)], rtol=1e-5)
        last = min(t + 3, 6)
        np.testing.assert_allclose(buf.next_obs[t], [last, 10 * last])


def test_dqn_learns_a_contextual_bandit():
    """Reward equals one observation feature per action; greedy actions should pick the largest."""
    torch.set_num_threads(1)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    cfg = DQNConfig(arch="flat", hidden=128, lr=1e-3, batch_size=128)
    agent = DQNAgent(OBS, ACTIONS, N, cfg, seed=0)
    for _ in range(800):
        obs = rng.random((128, OBS)).astype(np.float32)
        act = rng.integers(0, ACTIONS, 128)
        agent.update({
            "obs": obs, "action": act, "reward": obs[np.arange(128), act].astype(np.float32),
            "next_obs": obs, "discount": np.zeros(128, np.float32), "next_mask": np.ones((128, ACTIONS), bool),
        })
    test = rng.random((500, OBS)).astype(np.float32)
    chosen = agent.q_values(test).argmax(axis=1)
    regret = test[:, :ACTIONS].max(axis=1) - test[np.arange(500), chosen]
    random_regret = (test[:, :ACTIONS].max(axis=1) - test[:, :ACTIONS].mean(axis=1)).mean()
    assert regret.mean() < 0.4 * random_regret
