from rlstudio.agent import base as agent_base
from rlstudio.environment import base as env_base
from rlstudio.experiment import base as exp_base
from rlstudio.experiment import record


class Experiment:
  """Defines an experiment."""
  def __init__(self,
               config: exp_base.Configuration,
               agent: agent_base.Agent):
    self.config = config
    self.agent = agent
    self.config.validate()

  def train(self, summary: record.Summary = None) -> None:
    """Trains the agent on training tasks and optionally records statistics in `summary`."""
    time = -1
    for round_id in range(self.config.repeat + 1):
      for task in self.config.train_tasks:
        print(f'Training agent for {self.config.train_episodes} episodes on task {task.id()}')
    
        for episode in range(self.config.train_episodes):
          # Evaluation.
          if not episode % self.config.train_eval_step:
            time += 1
            metadata = exp_base.EvaluationMetadata(time, task.id(), round_id, episode)
            returns = self._eval(metadata, summary, task)
            print(f'Episode {episode:4d}: Returns: {returns:.2f}')

          # Training
          timestep = task.reset()
          if self.config.reset_agent:
            self.agent.reset()

          while not timestep.last():
            decision = self.agent.decide(timestep, greedy=False)
            new_timestep = task.step(decision.action)
            self.agent.update(timestep, new_timestep)
            timestep = new_timestep

        # Post-training evaluation.
        time += 1
        metadata = exp_base.EvaluationMetadata(
          time, task.id(), round_id, self.config.train_episodes)
        returns = self._eval(metadata, summary, task)
        print(f'Final eval: Returns: {returns:.2f}')

  def test(self, summary: record.Summary = None) -> None:
    """Tests an agent on test tasks and optionally records statistics in `summary`."""
    if self.config.test_tasks is None:
      return

    time = -1
    for round_id in range(self.config.repeat + 1):
      for task_idx, task in self.config.test_tasks:
        for episode in range(self.config.test_episodes):
          time += 1
          metadata = exp_base.EvaluationMetadata(
            time, task.id(), round_id, episode_id)
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

    timestep = task.reset()
    if self.config.reset_agent:
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
