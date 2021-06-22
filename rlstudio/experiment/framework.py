from rlstudio.agent import base as agent_base
from rlstudio.environment import base as env_base
from rlstudio.experiment import base as exp_base
from rlstudio.stats import base as stats_base
from rlstudio.typing import RunId

import numpy as np
from typing import List


class Experiment:
  """Defines an experiment."""
  def __init__(self,
               run_id: RunId,
               config: exp_base.Configuration,
               agents: List[agent_base.Agent]):
    self.run_id = run_id
    self.config = config
    self.agents = agents
    self.config.validate()

  def train(self, summary: stats_base.Summary = None) -> None:
    """Trains the agent on training tasks and optionally records statistics in `summary`."""
    np.set_printoptions(precision=2)

    time = -1
    for round_id in range(self.config.repeat + 1):
      for task in self.config.train_tasks:
        print(f'Training agent for {self.config.train_episodes} episodes on task {task.id()}')
    
        for episode in range(self.config.train_episodes):
          # Evaluation.
          if not episode % self.config.train_eval_step:
            time += 1
            metadata = exp_base.EvaluationMetadata(
              self.run_id, time, task.id(), round_id, episode)
            returns = self._eval(metadata, summary, task)
            print(f'Episode {episode:4d}: Returns: {returns}')

          # Training
          timestep = task.reset()
          if self.config.reset_agent:
            for agent in self.agents:
              agent.reset()

          while not timestep.last():
            decisions = [agent.decide(timestep, greedy=False)
                         for agent in self.agents]
            new_timestep = task.step([d.action for d in decisions])
            for agent in self.agents:
              agent.update(timestep, new_timestep)
            timestep = new_timestep

        # Post-training evaluation.
        time += 1
        metadata = exp_base.EvaluationMetadata(
          self.run_id, time, task.id(), round_id, self.config.train_episodes)
        returns = self._eval(metadata, summary, task)
        print(f'Final eval: Returns: {returns}')

  def test(self, summary: stats_base.Summary = None) -> None:
    """Tests an agent on test tasks and optionally records statistics in `summary`."""
    if self.config.test_tasks is None:
      return

    np.set_printoptions(precision=2)
    time = -1
    for round_id in range(self.config.repeat + 1):
      for task_idx, task in self.config.test_tasks:
        for episode in range(self.config.test_episodes):
          time += 1
          metadata = exp_base.EvaluationMetadata(
            self.run_id, time, task.id(), round_id, episode)
          returns = self._eval(metadata, summary, task)
          print(f'Tested agent on task {task.id()}: Total return is {returns}')

  def _eval(self,
            metadata: exp_base.EvaluationMetadata,
            summary: stats_base.Summary,
            task: env_base.Task) -> float:
    """Evaluates the agent on a given task and records statistics.

    Note that, unlike training, during evaluation the agent takes
    a greedy action based on its current policy.
    """
    returns = np.zeros(shape=len(self.agents), dtype=np.float)

    timestep = task.reset()
    if self.config.reset_agent:
      for agent in self.agents:
        agent.reset()

      while not timestep.last():
        decisions = [agent.decide(timestep, greedy=True)
                     for agent in self.agents]
        if summary is not None:
          summary.record_decisions(metadata, timestep, decisions)

        timestep = task.step([d.action for d in decisions])
        returns += timestep.reward

    if summary is not None:
      summary.record_returns(metadata, returns)
      summary.commit_episode(metadata)

    return returns
