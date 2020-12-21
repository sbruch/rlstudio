import jax.numpy as jnp

"""Actions are normally represented with integers."""
Action = int

"""An `ndarray` representing a probability distribution."""
Distribution = jnp.ndarray

"""An 'ndarray' representing an embedding."""
Embedding = jnp.ndarray
PolicyEmbedding = Embedding
ValueEmbedding = Embedding
StateEmbedding = Embedding

"""Identifier of an environment.Task."""
TaskId = str

"""Identifier of an observation emitted by an `environment.Task`."""
ObservationId = str

"""Type of an observation emitted by an `environment.Task`."""
ObservationType = str

"""An 'ndarray` (typically a single float) representing a state value estimate."""
Value = jnp.ndarray


EpisodeId = int
Time = int
RoundId = int
