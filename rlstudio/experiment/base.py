from rlstudio.typing import EpisodeId, RoundId, TaskId, Time

from dataclasses import dataclass

@dataclass
class EvaluationMetadata:
  time: Time
  env_id: TaskId
  round_id: RoundId
  episode_id: EpisodeId
