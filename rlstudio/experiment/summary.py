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
  def finalize(self) -> None:
    """Finalizes the statistics recorded."""

  @abc.abstractmethod
  def serialize(self, output_dir: str) -> None:
    """Serializes the summary data into files."""

  @abc.abstractmethod
  def deserialize(self, input_dir: str) -> None:
    """Deserializes a Summary from data files."""
