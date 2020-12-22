from rlstudio.typing import EpisodeId, RoundId, TaskId, Time

from dataclasses import dataclass

@dataclass
class EvaluationMetadata:
  time: Time
  env_id: TaskId
  round_id: RoundId
  episode_id: EpisodeId


@dataclass
class Configuration:
  """Training and test configuration.

  repeat: Number of times to repeat the series of tasks.
  episodes: Number of (training and test) episodes per task.
  train_eval_step: Evaluation step size during training.
      Agent is evaluated every `eval_step` episodes of training.
  reset_agent: Whether to call `reset()` on the agent at the beginning of every episode.
  """
  repeat: int = 0
  episodes: int = 100
  train_eval_step: int = 1
  reset_agent: bool = True
