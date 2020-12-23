"""Abstraction of a statistics collector."""
from rlstudio.agent import base as agent_base
from rlstudio.environment import base as env_base
from rlstudio.experiment import base as exp_base

import abc


class Summary:
  """Records statistics from an experiment."""

  @abc.abstractmethod
  def record_return(self,
                    metadata: exp_base.EvaluationMetadata,
                    value: float) -> None:
    """Records a return."""

  @abc.abstractmethod
  def record_decision(self,
                      metadata: exp_base.EvaluationMetadata,
                      timestep: env_base.TimeStep,
                      decision: agent_base.Decision) -> None:
    """Records an agent's decision."""

  @abc.abstractmethod
  def commit_episode(self, metadata: exp_base.EvaluationMetadata) -> None:
    """Finalizes data for the preceding episode.."""
