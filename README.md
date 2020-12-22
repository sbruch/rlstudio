# RLStudio

This library sets up a basic framework to develop reinforcement learning agents,
define tasks, and create an experimentation protocol. An example use-case is shown
in the code below:

```python
from rlstudio.agent import actor_critic_rnn
from rlstudio.experiment import base, framework

task = ...  # An instace of rlstudio.environment.Task

agent = actor_critic_rnn.make(task.observation_spec(),
                              task.action_spec())

config = base.Configuration(train_tasks=[task])
exp = framework.Experiment(config, agent)
exp.train()
```
