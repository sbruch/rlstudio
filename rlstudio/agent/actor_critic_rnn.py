"""Implements an Asynchronous Advantage Actor-Critic agent with an RNN."""

from rlstudio.agent import base, buffer
from rlstudio.environment import base as env_base
from rlstudio.typing import Action, Distribution, PolicyEmbedding
from rlstudio.typing import StateEmbedding, SuccessorEmbedding, Value, ValueEmbedding

from dm_env import specs
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import rlax
from typing import Any, Callable, List, NamedTuple, Tuple

RNNState = Any
PolicyLogits = jnp.ndarray
Successor = jnp.ndarray
ModelOutput = Tuple[PolicyLogits, Value, Successor,
                    PolicyEmbedding, ValueEmbedding,
                    StateEmbedding, SuccessorEmbedding]
# Input to a PolicyValueNet consists of: Observation, previous reward, and previous action tensors.
PolicyValueNet = Callable[[List[jnp.ndarray], RNNState], Tuple[ModelOutput, RNNState]]


class AgentState(NamedTuple):
  """Holds network parameters, optimizer state, and RNN's internal state."""
  params: hk.Params
  opt_state: Any
  rnn_state: RNNState
  rnn_unroll_state: RNNState


class ActorCriticRNN(base.Agent):
  """Recurrent A3C agent."""
  def __init__(
      self,
      observation_spec: specs.Array,
      action_spec: specs.DiscreteArray,
      network: PolicyValueNet,
      initial_rnn_state: RNNState,
      optimizer: optax.GradientTransformation,
      rng: hk.PRNGSequence,
      buffer_length: int,
      discount: float,
      td_lambda: float,
      entropy_cost: float = 1.,
      critic_cost: float = 1.,
      successor_cost: float = 0.):
    @jax.jit
    def pack(trajectory: buffer.Trajectory) -> List[jnp.ndarray]:
      """Converts a trajectory into an input."""
      observations = trajectory.observations[:, None, ...]

      rewards = jnp.concatenate([trajectory.previous_reward,
                                 jnp.squeeze(trajectory.rewards, -1)], -1)
      rewards = jnp.squeeze(rewards)
      rewards = jnp.expand_dims(rewards, (1, 2))

      previous_action = jax.nn.one_hot(trajectory.previous_action, action_spec.num_values)
      actions = jax.nn.one_hot(jnp.squeeze(trajectory.actions, 1), action_spec.num_values)
      actions = jnp.expand_dims(jnp.concatenate([previous_action, actions], 0), 1)

      return [observations, rewards, actions]

    @jax.jit
    def loss(trajectory: buffer.Trajectory, rnn_unroll_state: RNNState):
      """"Computes a linear combination of the policy gradient loss and value loss
      and regularizes it with an entropy term."""
      inputs = pack(trajectory)

      # Dyanmically unroll the network. This Haiku utility function unpacks the
      # list of input tensors such that the i^{th} row from each input tensor
      # is presented to the i^{th} unrolled RNN module.
      (logits, values, successors, _, _, state_embeddings, _), new_rnn_unroll_state = hk.dynamic_unroll(
        network, inputs, rnn_unroll_state)
      trajectory_len = trajectory.actions.shape[0]

      # Compute the combined loss given the output of the model.
      td_errors = rlax.td_lambda(
        v_tm1=values[:-1, 0],
        r_t=jnp.squeeze(trajectory.rewards, -1),
        discount_t=trajectory.discounts * discount,
        v_t=values[1:, 0],
        lambda_=jnp.array(td_lambda))
      critic_loss = jnp.mean(td_errors**2)
      actor_loss = rlax.policy_gradient_loss(
        logits_t=logits[:-1, 0],
        a_t=jnp.squeeze(trajectory.actions, 1),
        adv_t=td_errors,
        w_t=jnp.ones(trajectory_len))
      entropy_loss = jnp.mean(
        rlax.entropy_loss(logits[:-1, 0], jnp.ones(trajectory_len)))

      # Compute cosine similarity between successor representations.
      u = jax.lax.stop_gradient(state_embeddings[1:]) + discount * jax.lax.stop_gradient(successors[1:])
      v = successors[:-1]
      dot = jnp.linalg.norm(u * v, ord=1, axis=-1)
      u_norm = jnp.linalg.norm(u, ord=2, axis=-1)
      v_norm = jnp.linalg.norm(v, ord=2, axis=-1)
      dissimilarity = -dot / (u_norm * v_norm)
      successor_loss = jnp.nanmean(jnp.nan_to_num(dissimilarity, posinf=jnp.nan, neginf=jnp.nan))

      combined_loss = (actor_loss +
                       critic_cost * critic_loss +
                       entropy_cost * entropy_loss +
                       successor_cost * successor_loss)

      return combined_loss, new_rnn_unroll_state

    # Transform the loss into a pure function.
    loss_fn = hk.without_apply_rng(hk.transform(loss, apply_rng=True)).apply

    # Define update function.
    @jax.jit
    def sgd_step(state: AgentState,
                 trajectory: buffer.Trajectory) -> AgentState:
      """Performs a step of SGD over a trajectory."""
      gradients, new_rnn_state = jax.grad(
        loss_fn, has_aux=True)(state.params, trajectory,
                               state.rnn_unroll_state)
      updates, new_opt_state = optimizer.update(gradients, state.opt_state)
      new_params = optax.apply_updates(state.params, updates)
      return state._replace(
        params=new_params,
        opt_state=new_opt_state,
        rnn_unroll_state=new_rnn_state)

    # Initialize network parameters.
    init, forward = hk.without_apply_rng(hk.transform(network, apply_rng=True))
    dummy_observation = jnp.zeros((1, *observation_spec.shape),
                                  dtype=observation_spec.dtype)
    dummy_reward = jnp.zeros((1, 1, 1))
    dummy_action = jnp.zeros((1, 1, action_spec.num_values))
    inputs = [dummy_observation, dummy_reward, dummy_action]
    initial_params = init(next(rng), inputs, initial_rnn_state)
    initial_opt_state = optimizer.init(initial_params)

    # Internalize state.
    self._state = AgentState(initial_params, initial_opt_state,
                             initial_rnn_state, initial_rnn_state)
    self._forward = jax.jit(forward)
    self._buffer = buffer.Buffer(observation_spec, action_spec, buffer_length)
    self._sgd_step = sgd_step
    self._rng = rng
    self._initial_rnn_state = initial_rnn_state
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
    (logits, value, successor, policy_embedding,
     value_embedding, state_embedding, successor_embedding), rnn_state = (
      self._forward(self._state.params, inputs, self._state.rnn_state))
    self._state = self._state._replace(rnn_state=rnn_state)

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
      state_embedding=state_embedding,
      successor_embedding=successor_embedding)

  def update(self,
             timestep: env_base.TimeStep,
             new_timestep: env_base.TimeStep):
    """Adds a transition to the trajectory buffer and periodically does SGD."""
    if new_timestep.last():
      self.reset()

    self._buffer.append(timestep, new_timestep)
    if self._buffer.full() or new_timestep.last():
      trajectory = self._buffer.drain()
      self._state = self._sgd_step(self._state, trajectory)

  def reset(self):
    """Resets the internal RNN state."""
    self._state = self._state._replace(rnn_state=self._initial_rnn_state)

def make(observation_spec: specs.Array,
         action_spec: specs.DiscreteArray,
         rnn_hidden_size: int = 32,
         encoding_hidden_size: List[int] = [256,128,64],
         buffer_length: int = 120,
         discount: float = .5,
         td_lambda: float = .9,
         entropy_cost: float = 1.,
         critic_cost: float = 1.,
         successor_cost: float = 0.,
         seed: int = 0):
  """Creates a default agent."""
  initial_rnn_state = jnp.zeros((1, rnn_hidden_size), dtype=jnp.float32)

  def network(inputs: List[jnp.ndarray], state) -> ModelOutput:
    observation = hk.Flatten()(inputs[0]).reshape((1, -1))
    previous_reward = inputs[1].reshape((1, 1))
    previous_action = inputs[2].reshape((1, -1))
    
    torso = hk.nets.MLP(encoding_hidden_size)
    gru = hk.GRU(rnn_hidden_size)
    policy_head = hk.Linear(action_spec.num_values)
    value_head = hk.Linear(1)
    successor_torso = hk.nets.MLP([rnn_hidden_size, rnn_hidden_size])
    successor_head = hk.Linear(rnn_hidden_size)

    input_embedding = jnp.concatenate([observation, previous_reward, previous_action], -1)
    input_embedding = torso(input_embedding)
    embedding, state = gru(input_embedding, state)
    logits = policy_head(embedding)
    value = value_head(embedding)
    successor_embedding = successor_torso(embedding)
    successor = successor_head(successor_embedding)

    return (logits, jnp.squeeze(value, axis=-1), successor,
            embedding, embedding, embedding, successor_embedding), state

  return ActorCriticRNN(
    observation_spec=observation_spec,
    action_spec=action_spec,
    network=network,
    initial_rnn_state=initial_rnn_state,
    optimizer=optax.rmsprop(1e-3),
    rng=hk.PRNGSequence(seed),
    buffer_length=buffer_length,
    discount=discount,
    td_lambda=td_lambda,
    entropy_cost=entropy_cost,
    critic_cost=critic_cost,
    successor_cost=successor_cost,)
