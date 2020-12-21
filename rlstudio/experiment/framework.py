from rlstudio.agent import base as agent_base
from rlstudio.environment import base as env_base
from rlstudio.experiment import base as exp_base
from rlstudio.experiment import record

from typing import List


class Experiment:
  """Defines an experiment."""
  def __init__(self, agent: agent_base.Agent):
    self.agent = agent

  def train(self,
            tasks: List[env_base.Task],
            repeat: int = 0,
            episodes: int = 100,
            eval_step: int = 1,
            summary: record.Summary = None) -> None:
    """Trains the agent on a series of tasks.

    Args:
      tasks: A list of tasks to train the agent on.
      repeat: Number of times to repeat the series of tasks.
      episodes: Number of training episodes per task.
      eval_step: Evaluation step size. The agent is evaluated every `eval_step` episodes.
      summary: An optional `Summary` module to record statistics.
    """
    if len(tasks) == 0:
      raise ValueError('At least one task must be provided to train the agent on')
    for task in tasks:
      if type(task) is not type(tasks[0]):
        raise ValueError('All tasks used for training must be of the same type')

    time = -1
    for round_id in range(repeat + 1):
      for task in tasks:
        print(f'Training agent for {episodes} episodes on task {task.id()}')
    
        for episode in range(episodes):
          # Evaluation.
          if not episode % eval_step:
            time += 1
            metadata = exp_base.EvaluationMetadata(time, task.id(), round_id, episode)
            returns = self._eval(metadata, summary, task)
            print(f'Episode {episode:4d}: Returns: {returns:.2f}')

          # Training
          timestep = task.reset()
          self.agent.reset()

          while not timestep.last():
            decision = self.agent.decide(timestep, greedy=False)
            new_timestep = task.step(decision.action)
            self.agent.update(timestep, new_timestep)
            timestep = new_timestep

        # Post-training evaluation.
        time += 1
        metadata = exp_base.EvaluationMetadata(time, task.id(), round_id, episodes)
        returns = self._eval(metadata, summary, task)
        print(f'Final eval: Returns: {returns:.2f}')

  def test(self,
           tasks: List[env_base.Task],
           summary: record.Summary = None) -> None:
    """Tests an agent on a set of tasks and optionally records statistics in `summary`."""
    if len(tasks) == 0:
      raise ValueError('At least one task must be provided to test the agent on')
    for task in tasks:
      if type(task) is not type(tasks[0]):
        raise ValueError('All tasks used for testing must be of the same type')
    
    for task_idx, task in tasks:
      metadata = exp_base.EvaluationMetadata(
        time=0, env_id=task.id(), round_id=0, episode_id=task_idx)
      returns = self._eval(metadata, summary, task)
      print(f'Tested agent on task {task.id()}: Total return is {returns:.2f}')

  def _eval(self,
            metadata: exp_base.EvaluationMetadata,
            summary: record.Summary,
            task: env_base.Task) -> float:
    """Evaluates the agent on a given task and records statistics.

    Note that, unlike training, during evaluation the agent takes
    a greedy action based on its current policy.
    """
    returns = 0.

    # Start a new episode and reset the initial state of the agent.
    timestep = task.reset()
    self.agent.reset()

    while not timestep.last():
      decision = self.agent.decide(timestep, greedy=True)
      if summary is not None:
        summary.record_decision(metadata, timestep, decision)
      timestep = task.step(decision.action)
      returns += timestep.reward

    if summary is not None:
      summary.record_return(metadata, returns)

    return returns
