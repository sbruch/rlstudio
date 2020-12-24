from rlstudio.environment import base as env_base
from rlstudio.experiment import base as exp_base
from rlstudio.typing import ObservationId, ObservationType, TaskId

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


class TaskEmbeddings:
  """Encapsulates state embeddings from a single episode of a single task."""
  def __init__(self,
               id: str,
               task_id: TaskId,
               observation_type: ObservationType,
               sorted_observation_ids: List[ObservationId],
               embedding_size: int):
    """Creates an `TaskEmbeddings` object.

    Args:
      id: An identifier.
      task_id: Task identifier.
      observation_type: Type of observation to record. All other types will be ignored.
      sorted_observation_ids: A sorted list of observations to record.
          Observations not included in this list will be dismissed.
      embedding_size: Size of the embedding.
    """
    self.id = id
    self.task_id = task_id
    self.observation_type = observation_type
    self.sorted_observation_ids = sorted_observation_ids
    self.embedding_size = embedding_size

    self.stats = np.zeros((len(sorted_observation_ids), embedding_size))
    self.counts = np.zeros((len(sorted_observation_ids), 1))

  def record(self,
             metadata: exp_base.EvaluationMetadata,
             timestep: env_base.TimeStep,
             embedding: np.ndarray) -> None:
    """Records a 1-dimensional embedding."""
    if (self.task_id != metadata.task_id or
        self.observation_type != timestep.observation_type or
        timestep.observation_id not in self.sorted_observation_ids):
      return

    embedding = np.squeeze(embedding)
    if embedding.ndim > 1 or len(embedding) != self.embedding_size:
      raise ValueError(f'Expected a 1D embedding of size {self.embedding_size} '
                       f'but got {embedding.shape}')
    s = self.sorted_observation_ids.index(timestep.observation_id)
    self.stats[s] += embedding
    self.counts[s][0] += 1

  def commit(self) -> None:
    """Finalizes the statistics."""
    self.stats /= self.counts
    self.counts[:] = np.nan

  def is_compatible(self, other) -> bool:
    if not isinstance(other, TaskEmbeddings):
      return False
    if not np.array_equal(self.sorted_observation_ids, other.sorted_observation_ids):
      return False
    if self.embedding_size != other.embedding_size:
      return False
    return True

  def similarity(self, other) -> np.ndarray:
    """Computes the pairwise centered cosine similarity with another `TaskEmbeddings`.
    
    Rows represent this object and columns `other`.
    """
    if not self.is_compatible(other):
      raise ValueError(f'Incompatible embeddings: {self.id} vs. {other.id}')

    dim = self.stats.shape[0]
    matrix = np.zeros((dim, dim))

    u = self.stats - np.mean(self.stats, axis=1, keepdims=True)
    v = other.stats - np.mean(other.stats, axis=1, keepdims=True)

    for i in range(dim):
      for j in range(dim):
        norm = np.linalg.norm(u[i]) * np.linalg.norm(v[j])
        if norm > 0:
          matrix[i][j] = np.dot(u[i], v[j]) / norm

    return matrix


class Embeddings:
  """State embeddings for multiple tasks."""
  def __init__(self, id: str,
               task_ids: List[TaskId],
               observation_type: ObservationType,
               sorted_observation_ids: List[ObservationId],
               embedding_size: int):
    """Creates an `Embeddings` object.

    Args:
      id: An identifier.
      task_ids: Task identifiers for which embeddings are to be recorded.
      observation_type: Type of observation to record. All other types will be ignored.
      sorted_observation_ids: A sorted list of observations to record.
          Observations not included in this list will be dismissed.
      embedding_size: Size of the embedding.
    """
    self.id = id
    self.task_ids = task_ids
    self.observation_type = observation_type
    self.sorted_observation_ids = sorted_observation_ids
    self.embedding_size = embedding_size

    self.dataset = {}
    for t in task_ids:
      self.dataset[t] = TaskEmbeddings(
          f'{t}_{id}', t, observation_type,
          sorted_observation_ids, embedding_size)

  def record(self,
             metadata: exp_base.EvaluationMetadata,
             timestep: env_base.TimeStep,
             embedding: np.ndarray) -> None:
    """Records a 1-dimensional embedding."""
    if (metadata.task_id not in self.dataset or
        self.observation_type != timestep.observation_type or
        timestep.observation_id not in self.sorted_observation_ids):
      return
    self.dataset[metadata.task_id].record(metadata, timestep, embedding)

  def commit(self, metadata: exp_base.EvaluationMetadata) -> None:
    """Finalizes the statistics."""
    if metadata.task_id in self.dataset:
      self.dataset[metadata.task_id].commit()

  def is_compatible(self, other) -> bool:
    if not isinstance(other, Embeddings):
      return False
    if not np.array_equal(self.task_ids, other.task_ids):
      return False
    for t, e in self.dataset:
      if not e.is_compatible(other.dataset[t]):
        return False
    return True


def render_similarity(matrix: np.ndarray,
                      labels: List[str],
                      xlabel: str, ylabel: str,
                      specials: Dict[int, str]=None):
  """Renders a pairwise similarity matrix as a heatmap.

  Args:
    matrix: A 2-dimensional similarity matrix.
    labels: Labels for each row/column.
    xlabel: Label for the x axis.
    ylabel: Label for the y axis.
    specials: Dictionary containing row/column index to mark as keys,
        mapped to a color.

  Returns:
    A dictionary from plot identifier to a tuple containing
        `matplotlib.figure.Figure` and `matplotlib.axes.Axes`.
  """
  if matrix.ndim < 2 or matrix.shape[0] != matrix.shape[1]:
    raise ValueError(f'Invalid similarity matrix of shape {matrix.shape}')
  if len(labels) != matrix.shape[0]:
    raise ValueError(f'Expected {matrix.shape[0]} labels but got {len(labels)}')

  fig, ax = plt.subplots(figsize=(8, 6))
  imshow = ax.imshow(matrix, interpolation='none',
                     cmap='RdBu', origin='upper',
                     vmin=-1., vmax=1., aspect='auto')

  # Add ticks.
  ticklabels, indices = np.unique(labels, return_index=True)
  indices = sorted(indices)
  ticklabels = [labels[t] for t in indices]

  dividers = [d - .5 for d in indices]
  gaps = np.append(dividers, len(labels) - .5)
  gaps = (gaps[1:] - gaps[:-1]) / 2.
  ticks = dividers + gaps

  ax.tick_params(axis=u'both', which=u'both', length=0, pad=15)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_yticks(ticks)
  ax.set_xticks(ticks)
  ax.set_yticklabels(ticklabels, ha='center')
  ax.set_xticklabels(ticklabels, ha='center')

  for position in dividers:
    ax.axhline(position, color='k', linestyle='-', linewidth=.2)
    ax.axvline(position, color='k', linestyle='-', linewidth=.2)

  # Add markers if requested.
  if specials is not None:
    for idx, color in specials.items():
      ax.add_artist(matplotlib.patches.Ellipse(
          (-.6, idx), width=.3, height=.3,
          color=color, clip_on=False))
      ax.add_artist(matplotlib.patches.Ellipse(
          (idx, matrix.shape[0] - .35), width=.3, height=.3,
          color=color, clip_on=False))

  fig.colorbar(imshow, shrink=.3)
  plt.tight_layout()
  plt.close()

  return fig, ax
