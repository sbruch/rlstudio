"""Abstraction of an environment."""

from rlstudio.typing import Action, ObservationId, ObservationType, TaskId

import abc
from collections import namedtuple
import dm_env


class TimeStep(
    namedtuple('TimeStep', dm_env.TimeStep._fields +
               ('observation_id', 'observation_type', 'action')),
    dm_env.TimeStep):
  """Extends `dm_env.TimeStep` with additional attributes and metadata to
  represent a complete transition.

  In addition to `step_type`, `observation`, `reward`, and `discount`, this
  extended class also encapsulates the following properties of a transition:

  observation_id: A `str` identifier for the observation presented to the agent.
  observation_type: The type of observation presented to the agent.
  action: The `Action` the agent took to arrive at the current observation.
  """
  pass


class Task(dm_env.Environment):
  """An extension of `dm_env.Environment`."""

  @abc.abstractmethod
  def id(self) -> TaskId:
    """Id of this environment."""


def restart(observation,
            observation_id: ObservationId,
            observation_type: ObservationType) -> TimeStep:
  """Begins a fresh episode."""
  return TimeStep(
    step_type=dm_env.StepType.FIRST,
    reward=None,
    discount=None,
    observation=observation,
    observation_id=observation_id,
    observation_type=observation_type,
    action=None)

def transition(action: Action, reward, observation,
               discount=1.0,
               observation_id: ObservationId=None,
               observation_type: ObservationType=None):
  """Creates a transition."""
  return TimeStep(
    step_type=dm_env.StepType.MID,
    reward=reward,
    discount=discount,
    observation=observation,
    observation_id=observation_id,
    observation_type=observation_type,
    action=action)

def terminate(action: Action, reward, observation):
  """Terminates the current episode."""
  return TimeStep(
    step_type=dm_env.StepType.LAST,
    reward=reward,
    discount=0.,
    observation=observation,
    observation_id=None,
    observation_type=None,
    action=action)
