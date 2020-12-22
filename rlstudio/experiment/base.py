from rlstudio.environment import base as env_base
from rlstudio.typing import EpisodeId, RoundId, TaskId, Time

from dataclasses import dataclass
from typing import List


@dataclass
class EvaluationMetadata:
  time: Time
  env_id: TaskId
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
  train_tasks: List[env_base.Task],
  train_episodes: int = 100
  train_eval_step: int = 1

  test_tasks: List[env_base.Task],
  test_episodes: int = 1

  repeat: int = 0
  reset_agent: bool = True

  def validate(self):
    """Validates the configuration, raising an exception if invalid."""
    if len(train_tasks) == 0:
      raise ValueError('At least one task must be provided to train the agent on')
    for task in train_tasks:
      if type(task) is not type(train_tasks[0]):
        raise ValueError('All tasks used for training must be of the same type')

    for task in test_tasks:
      if type(task) is not type(test_tasks[0]):
        raise ValueError('All tasks used for testing must be of the same type')
