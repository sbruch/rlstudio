from rlstudio.environment import base as env_base
from rlstudio.typing import EpisodeId, RoundId, RunId, TaskId, Time

from dataclasses import dataclass
import numpy as np
from typing import List


@dataclass
class EvaluationMetadata:
  run_id: RunId
  time: Time
  task_id: TaskId
  round_id: RoundId
  episode_id: EpisodeId


@dataclass
class Configuration:
  """Training and test configuration.

  train_tasks: Series of tasks for training.
  train_episodes: Number of training episodes per task.
  train_eval_step: Evaluation step size during training.
      Agent is evaluated every `eval_step` episodes of training.

  test_tasks: Series of tasks for testing.
  test_episodes: Number of test episodes per task.

  repeat: Number of times to repeat the series of tasks.
  reset_agent: Whether to call `reset()` on the agent at the beginning of every episode.
  """
  train_tasks: List[env_base.Task]
  train_episodes: int = 100
  train_eval_step: int = 1

  test_tasks: List[env_base.Task] = None
  test_episodes: int = 1

  repeat: int = 0
  reset_agent: bool = True

  def validate(self):
    """Validates the configuration, raising an exception if invalid."""
    if len(self.train_tasks) == 0:
      raise ValueError('At least one task must be provided to train the agent on')
    for task in self.train_tasks:
      if type(task) is not type(self.train_tasks[0]):
        raise ValueError('All tasks used for training must be of the same type')

    if self.test_tasks is not None:
      for task in self.test_tasks:
        if type(task) is not type(self.test_tasks[0]):
          raise ValueError('All tasks used for testing must be of the same type')

  def __eq__(self, other):
    """Determines if a given `Configuration` object is equal to this."""
    if not isinstance(other, Configuration):
      return False
    if not np.array_equal(self.train_tasks, other.train_tasks):
      return False
    if not np.array_equal(self.test_tasks, other.test_tasks):
      return False

    if (self.train_episodes != other.train_episodes or
        self.train_eval_step != other.train_eval_step or
        self.test_episodes != other.test_episodes or
        self.repeat != other.repeat or
        self.reset_agent != other.reset_agent):
      return False

    return True
