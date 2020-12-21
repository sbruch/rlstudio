"""Abstraction of an agent."""

from rlstudio.environment import base as env_base
from rlstudio.typing import Action, Distribution, PolicyEmbedding
from rlstudio.typing import StateEmbedding, Value, ValueEmbedding

import abc
from dataclasses import dataclass


@dataclass
class Decision:
  """Encapsulates data explaining a decision.

  action: An action chosen by the agent.
  action_dist: The probability distribution over all actions.
  policy_embedding: Embeddings from the agent's policy function.
  value: State value estimates.
  value_embedding: Embeddings from the agent's value function.
  state_embedding: Embeddings representing a state.
  """
  action: Action
  action_dist: Distribution
  policy_embedding: PolicyEmbedding
  value: Value
  value_embedding: ValueEmbedding
  state_embedding: StateEmbedding


class Agent(abc.ABC):
  """An agent consists of an action-selection mechanism and an update rule."""

  @abc.abstractmethod
  def decide(self, ts: env_base.TimeStep, greedy: bool) -> Decision:
    """Chooses an action according to the agent's policy and returns metadata.
    
    Args:
      ts: A `Timestep` object describing the current time step.
      greedy: Whether to select actions according to a greedy policy.
    """

  @abc.abstractmethod
  def reset(self) -> None:
    """Resets the agent."""

  @abc.abstractmethod
  def update(self,
             previous: env_base.TimeStep,
             current: env_base.TimeStep) -> None:
    """Updates the agent given the previous and current time steps."""
