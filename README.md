# RLStudio

This library sets up a basic framework to develop reinforcement learning agents,
define tasks, and create an experimentation protocol. An example use-case is shown
in the code below:

```python
from rlstudio.agent import actor_critic_rnn
from rlstudio.experiment import framework

task = ...  # An instace of rlstudio.environment.Task

agent = actor_critic_rnn.make(
    task.observation_spec(), task.action_spec(),
    buffer_length=32, encoding_hidden_size=[64, 32],
    rnn_hidden_size=32, discount=.9,
    entropy_cost=1., critic_cost=1., seed=0)
exp = framework.Experiment(agent=agent)
exp.train(tasks=[task])
```
