from rlstudio.environment import base

from dataclasses import dataclass
from dm_env import specs
import numpy as np
from typing import NamedTuple


class Trajectory(NamedTuple):
  """A trajectory is a sequence of observations, actions, rewards, discounts."""
  previous_reward: np.ndarray  # [num_agents] Holds reward at time -1
  previous_action: np.ndarray  # [num_agents] Holds action at time -1
  observations: np.ndarray  # [T + 1, ...]
  actions: np.ndarray  # [T, num_agents]
  rewards: np.ndarray  # [T, num_agents]
  discounts: np.ndarray  # [T]


class Buffer:
  """A buffer that accumulates trajectories."""

  _previous_reward: np.ndarray
  _previous_action: np.ndarray
  _observations: np.ndarray
  _actions: np.ndarray
  _rewards: np.ndarray
  _discounts: np.ndarray

  _max_trajectory_length: int
  _needs_reset: bool = True
  _t: int = 0

  def __init__(
      self,
      observation_spec: specs.Array,
      action_spec: specs.Array,
      max_trajectory_length: int):
    """Allocates capacity to hold a trajectory."""
    self.num_agents: int = 1
    if isinstance(action_spec, list):
      self.num_agents = len(action_spec)

    self._previous_reward = np.zeros(shape=(self.num_agents), dtype=np.float32)
    self._previous_action = -1. * np.ones(shape=(self.num_agents), dtype=np.float32)
    self._observations = np.zeros(
      shape=(max_trajectory_length + 1, *observation_spec.shape),
      dtype=observation_spec.dtype)
    self._actions = np.zeros(
      shape=(max_trajectory_length, self.num_agents, *action_spec.shape),
      dtype=action_spec.dtype)
    self._rewards = np.zeros(shape=(max_trajectory_length, self.num_agents), dtype=np.float32)
    self._discounts = np.zeros(max_trajectory_length, dtype=np.float32)
    self._max_trajectory_length = max_trajectory_length

  def append(self,
             timestep: base.TimeStep,
             new_timestep: base.TimeStep):
    """Appends a transition to the buffer."""
    if self.full():
      raise ValueError('Buffer is full.')

    # Start a new trajectory with an initial observation, if required.
    # Also keep track of previous action and reward.
    if self._needs_reset:
      self._t = 0
      if timestep.reward is None:
        self._previous_reward = np.zeros(shape=(self.num_agents), dtype=np.float32)
      else:
        r = np.array(timestep.reward)
        if len(r.shape) == 0:
          r = np.expand_dims(r, 0)
        self._previous_reward = r

      if timestep.action is None:
        self._previous_action = -1. * np.ones(shape=(self.num_agents), dtype=np.float32)
      else:
        a = np.array(timestep.action)
        if len(a.shape) == 0:
          a = np.expand_dims(a, 0)
        self._previous_action = a

      self._observations[self._t] = timestep.observation
      self._needs_reset = False

    # Append to the buffer.
    self._observations[self._t + 1] = new_timestep.observation
    self._actions[self._t] = new_timestep.action
    self._rewards[self._t] = new_timestep.reward
    self._discounts[self._t] = new_timestep.discount
    self._t += 1

    if new_timestep.last():
      self._needs_reset = True

  def drain(self) -> Trajectory:
    """Empties the buffer and returns a trajectory."""
    if self.empty():
      raise ValueError('Buffer is empty.')
    trajectory = Trajectory(
      self._previous_reward,
      self._previous_action,
      self._observations[:self._t + 1],
      self._actions[:self._t],
      self._rewards[:self._t],
      self._discounts[:self._t],
    )
    self._t = 0
    self._needs_reset = True
    return trajectory

  def empty(self) -> bool:
    """Returns whether or not the trajectory buffer is empty."""
    return self._t == 0

  def full(self) -> bool:
    """Returns whether or not the trajectory buffer is full."""
    return self._t == self._max_trajectory_length
