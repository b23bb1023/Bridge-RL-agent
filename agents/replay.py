"""Replay memory with n-step returns and stored next-state action masks."""

from __future__ import annotations

from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, n_actions: int, seed: int = 0, reward_dim: int = 1) -> None:
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity if reward_dim == 1 else (capacity, reward_dim), dtype=np.float32)
        self.discount = np.zeros(capacity, dtype=np.float32)
        self.next_mask = np.zeros((capacity, n_actions), dtype=bool)
        self.mask = np.ones((capacity, n_actions), dtype=bool)
        self.next_action = np.zeros(capacity, dtype=np.int64)
        self.has_next_action = False   # set when transitions record the evaluated policy's next action
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, discount, next_mask, mask=None, next_action=None) -> None:
        i = self.ptr
        self.obs[i] = obs
        if mask is not None:
            self.mask[i] = mask
        if next_action is not None:
            self.next_action[i] = next_action
            self.has_next_action = True
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.discount[i] = discount
        self.next_mask[i] = next_mask
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_obs": self.next_obs[idx],
            "discount": self.discount[idx],
            "next_mask": self.next_mask[idx],
            "mask": self.mask[idx],
            **({"next_action": self.next_action[idx]} if self.has_next_action else {}),
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional prioritised replay on a sum-tree; sampling and updates are vectorised over the batch."""

    def __init__(self, capacity: int, obs_dim: int, n_actions: int, alpha: float = 0.6, seed: int = 0, reward_dim: int = 1) -> None:
        super().__init__(capacity, obs_dim, n_actions, seed, reward_dim)
        self.alpha = alpha
        self.leaves = 1 << int(np.ceil(np.log2(max(2, capacity))))
        self.tree = np.zeros(2 * self.leaves, dtype=np.float64)
        self.max_priority = 1.0

    def _set(self, idx: np.ndarray, priority: np.ndarray) -> None:
        pos = idx + self.leaves
        self.tree[pos] = priority
        pos //= 2
        while pos[0] >= 1:
            pos = np.unique(pos)
            self.tree[pos] = self.tree[2 * pos] + self.tree[2 * pos + 1]
            pos //= 2

    def add(self, obs, action, reward, next_obs, discount, next_mask, mask=None) -> None:
        i = self.ptr
        super().add(obs, action, reward, next_obs, discount, next_mask, mask)
        self._set(np.array([i]), np.array([self.max_priority]))

    def sample(self, batch_size: int, beta: float = 0.4) -> dict[str, np.ndarray]:
        total = self.tree[1]
        targets = (np.arange(batch_size) + self.rng.random(batch_size)) * (total / batch_size)
        pos = np.ones(batch_size, dtype=np.int64)
        while pos[0] < self.leaves:
            left = 2 * pos
            go_right = targets > self.tree[left]
            targets = np.where(go_right, targets - self.tree[left], targets)
            pos = np.where(go_right, left + 1, left)
        idx = np.minimum(pos - self.leaves, self.size - 1)
        probs = self.tree[idx + self.leaves] / total
        weights = (self.size * np.maximum(probs, 1e-12)) ** (-beta)
        batch = {
            "obs": self.obs[idx], "action": self.action[idx], "reward": self.reward[idx],
            "next_obs": self.next_obs[idx], "discount": self.discount[idx], "next_mask": self.next_mask[idx],
            "weights": (weights / weights.max()).astype(np.float32), "indices": idx,
        }
        return batch

    def update_priorities(self, idx: np.ndarray, td_error: np.ndarray) -> None:
        priority = (np.abs(td_error) + 1e-3) ** self.alpha
        self.max_priority = max(self.max_priority, float(priority.max()))
        order = np.argsort(idx)
        idx, priority = idx[order], priority[order]
        keep = np.r_[idx[1:] != idx[:-1], True]   # one priority per leaf when a transition is drawn twice
        self._set(idx[keep], priority[keep])


class NStepCollector:
    """Turns one-step transitions into n-step ones: R = sum_k gamma^k r_k, bootstrapped with gamma^n."""

    def __init__(self, n: int, gamma: float, buffer: ReplayBuffer) -> None:
        self.n, self.gamma, self.buffer = n, gamma, buffer
        self.queue: deque = deque()

    def _emit(self, next_obs, next_mask, terminal: bool) -> None:
        obs, action, _, mask = self.queue[0]
        ret, g = 0.0, 1.0   # the reward may be a scalar or a per-bridge vector
        for _, _, r, _ in self.queue:
            ret += g * r
            g *= self.gamma
        self.buffer.add(obs, action, ret, next_obs, 0.0 if terminal else g, next_mask, mask)
        self.queue.popleft()

    def push(self, obs, action, reward, next_obs, next_mask, terminal: bool, mask=None) -> None:
        self.queue.append((obs, action, reward, mask))
        if terminal:
            while self.queue:
                self._emit(next_obs, next_mask, terminal=True)
        elif len(self.queue) == self.n:
            self._emit(next_obs, next_mask, terminal=False)
