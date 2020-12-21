"""Implements an Asynchronous Advantage Actor-Critic agent."""

from rlstudio.agent import base, buffer
from rlstudio.environment import base as env_base
from rlstudio.typing import Action, Distribution, PolicyEmbedding
from rlstudio.typing import StateEmbedding, Value, ValueEmbedding

from dataclasses import dataclass
from dm_env import specs
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import rlax
from typing import Any, Callable, List, NamedTuple, Tuple

Logits = jnp.ndarray
NetworkOutput = Tuple[Logits, Value, PolicyEmbedding, ValueEmbedding, StateEmbedding]
PolicyValueNet = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], NetworkOutput]


class AgentState(NamedTuple):
  """Holds network parameters and optimizer state."""
  params: hk.Params
  opt_state: Any


class ActorCritic(base.Agent):
  """Feed-forward actor-critic agent."""
  def __init__(
      self,
      observation_spec: specs.Array,
      action_spec: specs.DiscreteArray,
      network: PolicyValueNet,
      optimizer: optax.GradientTransformation,
      rng: hk.PRNGSequence,
      buffer_length: int,
      discount: float,
      td_lambda: float,
      entropy_cost: float = 1.,
      critic_cost: float = 1.):
    @jax.jit
    def pack(trajectory: buffer.Trajectory) -> List[jnp.ndarray]:
      """Converts a trajectory into an input."""
      observations = trajectory.observations[:, None, ...]

      rewards = jnp.concatenate([jnp.array([trajectory.previous_reward]),
                                 trajectory.rewards], -1)
      rewards = jnp.expand_dims(rewards, (1, 2))

      previous_action = jax.nn.one_hot([trajectory.previous_action], action_spec.num_values)
      actions = jax.nn.one_hot(trajectory.actions, action_spec.num_values)
      actions = jnp.expand_dims(jnp.concatenate([previous_action, actions], 0), 1)

      return observations, rewards, actions

    @jax.jit
    def loss(trajectory: buffer.Trajectory) -> jnp.ndarray:
      """"Actor-critic loss."""
      observations, rewards, actions = pack(trajectory)
      logits, values, _, _, _ = network(observations, rewards, actions)

      td_errors = rlax.td_lambda(
        v_tm1=values[:-1],
        r_t=trajectory.rewards,
        discount_t=trajectory.discounts * discount,
        v_t=values[1:],
        lambda_=jnp.array(td_lambda))
      critic_loss = jnp.mean(td_errors**2)
      actor_loss = rlax.policy_gradient_loss(
        logits_t=logits[:-1],
        a_t=trajectory.actions,
        adv_t=td_errors,
        w_t=jnp.ones_like(td_errors))

      entropy_loss = jnp.mean(
        rlax.entropy_loss(logits[:-1], jnp.ones_like(td_errors)))

      return actor_loss + critic_cost * critic_loss + entropy_cost * entropy_loss

    # Transform the loss into a pure function.
    loss_fn = hk.without_apply_rng(hk.transform(loss, apply_rng=True)).apply

    # Define update function.
    @jax.jit
    def sgd_step(state: AgentState,
                 trajectory: buffer.Trajectory) -> AgentState:
      """Performs a step of SGD over a trajectory."""
      gradients = jax.grad(loss_fn)(state.params, trajectory)
      updates, new_opt_state = optimizer.update(gradients, state.opt_state)
      new_params = optax.apply_updates(state.params, updates)
      return AgentState(params=new_params, opt_state=new_opt_state)

    # Initialize network parameters and optimiser state.
    init, forward = hk.without_apply_rng(hk.transform(network, apply_rng=True))
    dummy_observation = jnp.zeros((1, *observation_spec.shape),
                                  dtype=observation_spec.dtype)
    dummy_reward = jnp.zeros((1, 1, 1))
    dummy_action = jnp.zeros((1, 1, action_spec.num_values))
    initial_params = init(next(rng), dummy_observation, dummy_reward, dummy_action)
    initial_opt_state = optimizer.init(initial_params)

    # Internalize state.
    self._state = AgentState(initial_params, initial_opt_state)
    self._forward = jax.jit(forward)
    self._buffer = buffer.Buffer(observation_spec, action_spec, buffer_length)
    self._sgd_step = sgd_step
    self._rng = rng
    self._action_spec = action_spec

  def decide(self, timestep: env_base.TimeStep, greedy: bool=False) -> base.Decision:
    key = next(self._rng)
    previous_reward = timestep.reward if timestep.reward is not None else 0.
    previous_action = timestep.action if timestep.action is not None else -1
    inputs = [
      timestep.observation[None, ...],
      jnp.ones((1, 1, 1), dtype=jnp.float32) * previous_reward,
      jax.nn.one_hot([[previous_action]], self._action_spec.num_values)
    ]
    logits, value, policy_embedding, value_embedding, state_embedding = (
      self._forward(self._state.params, inputs[0], inputs[1], inputs[2]))
    if greedy:
      action = rlax.greedy().sample(key, logits).squeeze()
    else:
      action = jax.random.categorical(key, logits).squeeze()

    return base.Decision(
      action=int(action),
      action_dist=jax.nn.softmax(logits),
      policy_embedding=policy_embedding,
      value=value,
      value_embedding=value_embedding,
      state_embedding=state_embedding)

  def update(self,
             timestep: env_base.TimeStep,
             new_timestep: env_base.TimeStep):
    """Adds a transition to the trajectory buffer and periodically does SGD."""
    self._buffer.append(timestep, new_timestep)
    if self._buffer.full() or new_timestep.last():
      trajectory = self._buffer.drain()
      self._state = self._sgd_step(self._state, trajectory)

  def reset(self):
      pass


def make(observation_spec: specs.Array,
         action_spec: specs.DiscreteArray,
         hidden_size: List[int] = [256,128,64],
         buffer_length: int = 120,
         discount: float = .5,
         td_lambda: float = .9,
         entropy_cost: float = 1.,
         critic_cost: float = 1.,
         seed: int = 0):
  """Creates a default agent."""
  def network(observation: jnp.ndarray,
              previous_reward: jnp.ndarray,
              previous_action: jnp.ndarray) -> NetworkOutput:
    observation = hk.Flatten()(observation)
    previous_reward = hk.Flatten()(previous_reward)
    previous_action = hk.Flatten()(previous_action)

    torso = hk.nets.MLP(hidden_size)
    policy_head = hk.Linear(action_spec.num_values)
    value_head = hk.Linear(1)

    embedding = torso(jnp.concatenate([observation, previous_reward, previous_action], -1))
    logits = policy_head(embedding)
    value = value_head(embedding)
    return logits, jnp.squeeze(value, axis=-1), embedding, embedding, embedding

  return ActorCritic(
    observation_spec=observation_spec,
    action_spec=action_spec,
    network=network,
    optimizer=optax.rmsprop(1e-3),
    rng=hk.PRNGSequence(seed),
    buffer_length=buffer_length,
    discount=discount,
    td_lambda=td_lambda,
    entropy_cost=entropy_cost,
    critic_cost=critic_cost)
