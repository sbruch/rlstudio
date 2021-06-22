"""Abstraction of a statistics collector."""
from rlstudio.agent import base as agent_base
from rlstudio.environment import base as env_base
from rlstudio.experiment import base as exp_base

import abc
import numpy as np
from typing import List


class Summary:
  """Records statistics from an experiment."""

  @abc.abstractmethod
  def record_returns(self,
                     metadata: exp_base.EvaluationMetadata,
                     values: np.ndarray) -> None:
    """Records returns."""

  @abc.abstractmethod
  def record_decisions(self,
                       metadata: exp_base.EvaluationMetadata,
                       timestep: env_base.TimeStep,
                       decisions: List[agent_base.Decision]) -> None:
    """Records an agent's decision."""

  @abc.abstractmethod
  def commit_episode(self, metadata: exp_base.EvaluationMetadata) -> None:
    """Finalizes data for the preceding episode.."""
