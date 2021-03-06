from rlstudio.environment import base as env_base
from rlstudio.experiment import base as exp_base
from rlstudio.typing import ObservationId, ObservationType, TaskId

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rcca
from sklearn import decomposition, manifold, metrics
from typing import Dict, List

EmbeddingCollection=List['EmbeddingMatrix']


class EmbeddingMatrix:
  """An embedding matrix where each row is the representation of an item."""
  def __init__(self, items: int, dim: int):
    self.items = items
    self.dim = dim
    self.matrix = np.zeros((items, dim))

  def is_compatible(self, other) -> bool:
    if not isinstance(other, EmbeddingMatrix):
      return False
    if self.items != other.items or self.dim != other.dim:
      return False
    return True

  def similarity(self, other) -> np.ndarray:
    """Computes the pairwise centered cosine similarity with another `EmbeddingMatrix`.

    Rows represent this object and columns `other`.
    """
    if not self.is_compatible(other):
      raise ValueError(f'Incompatible embedding matrices')
    matrix = np.zeros((self.items, self.items))

    u = self.matrix - np.mean(self.matrix, axis=1, keepdims=True)
    v = other.matrix - np.mean(other.matrix, axis=1, keepdims=True)

    for i in range(self.items):
      for j in range(self.items):
        norm = np.linalg.norm(u[i]) * np.linalg.norm(v[j])
        if norm > 0:
          matrix[i][j] = np.dot(u[i], v[j]) / norm

    return matrix


  def tsne(self):
    """Applies TSNE to the embeddings and returns coordinates."""
    projection = manifold.TSNE(n_components=2, metric="precomputed", square_distances=True)
    distance_matrix = metrics.pairwise_distances(self.matrix, self.matrix, metric='cosine')
    return projection.fit_transform(distance_matrix)


  def render_tsne(self,
                  coords: np.ndarray,
                  labels: List[str],
                  markers: List[str]=None,
                  filled: List[bool]=None,
                  axes=None):
    """Renders the embeddings using TSNE.

    Args:
      coords: 2D list of coordinates.
      labels: List of labels for every item.
      markers: List of markers to use. Default is 'o' if None.
      filled: List of booleans indicating whether the marker is filled.
          Default to True if None.
      axes: Optional `matplotlib.axes.Axes` to draw on.

    Returns:
      A tuple containing `matplotlib.figure.Figure` and `matplotlib.axes.Axes`.
    """
    if len(labels) != self.items:
      raise ValueError(f'Expected {self.items} labels but got {len(labels)}')

    x = coords[:, 0]
    y = coords[:, 1]

    labels = np.array(labels)
    unique_labels = np.unique(labels)
    cmap = iter(plt.cm.rainbow(np.linspace(0, 1, len(unique_labels))))
    colors = {}
    for l in unique_labels:
      colors[l] = next(cmap)

    if markers is None:
      markers = np.tile(['o'], len(labels))
    if filled is None:
      filled = np.tile([True], len(labels))

    axes_given = True
    if axes is None:
      fig, axes = plt.subplots()
      axes_given = False

    for i in range(self.items):
      axes.scatter(x[i], y[i],
                   facecolors=colors[labels[i]] if filled[i] else 'none',
                   edgecolors=colors[labels[i]],
                   marker=markers[i], s=100, label=labels[i])

    handles, labels = axes.get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    axes.legend(handles, labels, bbox_to_anchor=(-.35, 1.0), loc='upper left')

    axes.set_xticks([])
    axes.set_yticks([])

    if axes_given:
      return

    plt.tight_layout()
    plt.close()
    return fig, axes


def _apply_component_analysis(
    training: EmbeddingCollection,
    test: EmbeddingCollection,
    ncomponents: int,
    analysis='pca'):
  # Validate the arguments.
  if len(training) == 0:
    raise ValueError(f'No embedding matrix found for training')
  for em in training[1:]:
    if not training[0].is_compatible(em):
      raise ValueError(f'Training embedding matrices are not compatible with each other')
  if test is not None and len(test) != 0:
    for em in test:
      if training[0].dim != em.dim:
        raise ValueError(f'Test embedding matrices are not compatible with training matrices')

  ncomponents = min(ncomponents, training[0].dim, training[0].items)

  def _apply(collection, transform_fn):
    # Prepare datasets for analysis.
    items = collection[0].items
    matrix = np.zeros((len(collection) * items, collection[0].dim))
    for i, em in enumerate(collection):
      matrix[i*items:(i+1)*items] = em.matrix

    # Apply analysis.
    components = transform_fn(matrix)

    # Create a new collection of EmbeddingMatrix objects.
    comp_collection: EmeddingCollection = []
    for i in range(len(collection)):
      em = EmbeddingMatrix(items, components.shape[-1])
      em.matrix = components[i*items:(i+1)*items]
      comp_collection.append(em)

    return comp_collection

  # Training data.
  if analysis == 'ica':
    engine = decomposition.FastICA(n_components=ncomponents)
  else:
    engine = decomposition.PCA(n_components=ncomponents)

  training_comp_collection = _apply(training, engine.fit_transform)

  if test is None or len(test) == 0:
    return training_comp_collection, None, engine

  # Test data.
  test_comp_collection = _apply(test, engine.transform)
  return training_comp_collection, test_comp_collection, engine


def pca(
    training: EmbeddingCollection,
    test: EmbeddingCollection,
    ncomponents: int):
  """Applies PCA to extract principal components.

  Args:
    training: A collection of `EmbeddingMatrix` objects used to train PCA.
    test: A collection of `EmbeddingMatrix` objects used for testing.
    ncomponents: Number of components (>=1) or percentage of variance (<1) to keep.

  Returns:
    A tuple consisting of:
      - EmbeddingCollection containing principal components for the training sets.
      - None or EmbeddingCollection containing principal components for the test sets.
      - A `decomposition.PCA` object.
  """
  return _apply_component_analysis(training, test, ncomponents, 'pca')


def ica(
    training: EmbeddingCollection,
    test: EmbeddingCollection,
    ncomponents: int):
  """Applies ICA to extract principal components.

  Args:
    training: A collection of `EmbeddingMatrix` objects used to train PCA.
    test: A collection of `EmbeddingMatrix` objects used for testing.
    ncomponents: Number of components (>=1) or percentage of variance (<1) to keep.

  Returns:
    A tuple consisting of:
      - EmbeddingCollection containing principal components for the training sets.
      - None or EmbeddingCollection containing principal components for the test sets.
      - A `decomposition.ICA` object.
  """
  return _apply_component_analysis(training, test, ncomponents, 'ica')


def _cca_pack(collections: List[EmbeddingCollection]):
  """Prepare datasets for CCA."""
  dataset = []
  items = collections[0][0].items
  for collection in collections:
    matrix = np.zeros((len(collection) * items, collection[0].dim))
    for i, em in enumerate(collection):
      matrix[i*items:(i+1)*items] = em.matrix
    matrix -= np.mean(matrix, axis=0)
    dataset.append(matrix)
  return dataset, items

def _cca_unpack(collections, ccs, items):
  """Creates a new collection of EmbeddingMatrix objects."""
  cc_collections: List[EmeddingCollection] = []
  for i in range(len(collections)):
    cc_collection: EmbeddingCollection = []
    for j in range(len(collections[i])):
      em = EmbeddingMatrix(items, ccs[i].shape[-1])
      em.matrix = ccs[i][j*items:(j+1)*items]
      cc_collection.append(em)
    cc_collections.append(cc_collection)
  return cc_collections



def cca(
    training: List[EmbeddingCollection],
    test: List[EmbeddingCollection],
    ncomponents: int,
    reg=.01,
    verbose=False):
  """Applies CCA to extract canonical components.

  Args:
    training: A list where each item is a collection of `EmbeddingMatrix` objects used to train CCA.
    test: A list of the same size as `training` but where the collections are used for testing.
    ncomponents: Number of canonical components.
    reg: Regularization parameter.
    verbose: Sets the pyrcca.CCA's verbosity.

  Returns:
    A tuple consisting of:
      - List[EmbeddingCollection] containing principal components for the training sets.
      - None or List[EmbeddingCollection] containing principal components for the test sets.
      - An `rcca.CCA` object.
  """
  # Validate the arguments.
  if len(training) < 2:
    raise ValueError(f'Expected at least 2 training collections')
  reference = training[0]
  for collection in training[1:]:
    if len(collection) != len(reference):
      raise ValueError(f'Number of embedding matrices within each collection is not consistent in the training set')
    for em in collection:
      if em.items != reference[0].items:
        raise ValueError(f'Training embedding matrices do not have the same number of items')
  if test is not None and len(test) != 0:
    if len(test) != len(training):
      raise ValueError('Number of collections must be the same in the training and test sets')
    for collection in test:
      for em in collection:
        if em.items != test[0][0].items:
          raise ValueError(f'Test embedding matrices do not have the same number of items')

  _cca = rcca.CCA(kernelcca=False, reg=reg, numCC=ncomponents, verbose=verbose)
  # Training.
  training_set, items = _cca_pack(training)
  _cca.train(training_set)
  training_cc_collection = _cca_unpack(training, _cca.comps, items)

  if test is None or len(test) == 0:
    return training_cc_collection, None, _cca

  # Test.
  test_set, items = _cca_pack(test)
  _, _, comps = rcca.recon(test_set, _cca.ws, kernelcca=False)
  comps = np.array(comps)
  test_cc_collection = _cca_unpack(test, comps, items)
  return training_cc_collection, test_cc_collection, _cca


def cca_predict(
    cca: rcca.CCA,
    sets: List[EmbeddingCollection],
    cutoff=1e-15):
  """Transforms the embedding collections accorind go the given `cca` weights.

  Args:
    cca: CCA object.
    sets: Input matrices to transform.

  Returns:
    List[EmbeddingCollection] consisting of transformed embedding matrices.
  """
  def _listdot(d1, d2): return [np.dot(x[0].T, x[1]) for x in zip(d1, d2)]

  dataset, items = _cca_pack(sets)

  iws = [np.linalg.pinv(w.T, rcond=cutoff) for w in cca.ws]
  ccomp = _listdot([d.T for d in dataset], cca.ws)
  ccomp = np.array(ccomp)
  preds = []

  for dnum in range(len(dataset)):
    idx = np.ones((len(dataset),))
    idx[dnum] = False
    proj = ccomp[idx > 0].mean(0)
    pred = np.dot(iws[dnum], proj.T).T
    preds.append(pred)

  return _cca_unpack(dataset, preds, items)


def render_components(
    matrices: Dict[str, EmbeddingCollection],
    ncomponents: int,
    labels: List[str],
    xlabel: str, ylabel: str,
    specials: Dict[int, str]=None):
  """Renders the first `ncomponents` components of the embedding matrices.

  Args:
    matrices: A mapping from an identifier to a collection of `EmbeddingMatrix`.
        All embedding matrices (within and across identifiers) must be compatible (i.e.,
        have equal dimension and number of items).
        For every identifier, the component value is averaged over its collection of embedding matrices.
    ncomponents: The number of components to render.
    labels: List of labels for every row of the matrix.
    xlabel: Label for the x axis.
    ylabel: Label for the y axis.
    specials: Mapping from the index of special rows to a color, to be marked in the final plot.

  Returns:
    A dictionary from plot identifier to a tuple containing
        `matplotlib.figure.Figure` and `matplotlib.axes.Axes`.
  """
  # Validate the arguments.
  for id, ems in matrices.items():
    if len(ems) == 0:
      raise ValueError(f'No embedding matrix found for {id}')
    for em in ems[1:]:
      if not ems[0].is_compatible(em):
        raise ValueError(f'Embedding matrices for "{id}" are not compatible with each other')
    if len(labels) != ems[0].items:
      raise ValueError(f'Expected {ems[0].items} labels but got {len(labels)}')

    ncomponents = min(ncomponents, ems[0].dim)

  # Find unique labels.
  labels = np.array(labels)
  _, indices = np.unique(labels, return_index=True)
  unique_labels = [labels[i] for i in indices]

  dividers = [d - .5 for d in indices]
  gaps = np.append(dividers, len(labels) - .5)
  gaps = (gaps[1:] - gaps[:-1]) / 2.
  xticks = dividers + gaps

  # Build a color map.
  cmap = iter(plt.cm.PiYG(np.linspace(0, 1, len(matrices))))
  colors = {}
  for id in matrices.keys():
    colors[id] = next(cmap)

  # Generate figures.
  plt.rcParams.update({'font.size': 20})
  figures = {}
  for component in range(ncomponents):
    fig, ax = plt.subplots(figsize=(8, 6))

    for id, ems in matrices.items():
      data = np.expand_dims(ems[0].matrix[:, component], axis=1)
      for em in ems[1:]:
        data = np.concatenate(
          [data, np.expand_dims(em.matrix[:, component], axis=1)],
          axis=1)
      y = np.mean(data, axis=1)
      std = np.std(data, axis=1)

      ax.plot(np.arange(y.shape[0]), y, label=id, c=colors[id])
      ax.fill_between(np.arange(y.shape[0]), y - std, y + std,
                      color=colors[id], alpha=.2)

    ax.set_xlim(-.5, len(labels))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xticks)
    ax.set_xticklabels(unique_labels, ha='center')
    ax.tick_params(axis=u'x', which=u'both', length=0, pad=15)
    for position in dividers:
      ax.axvline(position, color='k', linestyle=':', linewidth=.1)

    if specials is not None:
      ymin, ymax = ax.get_ylim()
      for x, c in specials.items():
        ax.add_artist(matplotlib.patches.Ellipse(
          (x, ymin), width=.5, height=(ymax-ymin)/50.,
          color=c, clip_on=False))

    plt.legend(loc='best')
    plt.tight_layout()
    plt.close()
    figures[f'{component}'] = (fig, ax)

  return figures


class TaskEmbeddings(EmbeddingMatrix):
  """Encapsulates state embeddings from a single episode of a single task."""
  def __init__(self,
               id: str,
               task_id: TaskId,
               observation_type: ObservationType,
               sorted_observation_ids: List[ObservationId],
               embeddings_per_observation: int,
               embedding_size: int):
    """Creates an `TaskEmbeddings` object.

    Args:
      id: An identifier.
      task_id: Task identifier.
      observation_type: Type of observation to record. All other types will be ignored.
      sorted_observation_ids: A sorted list of observations to record.
          Observations not included in this list will be dismissed.
      embeddings_per_observation: Number of embeddings per observation id.
      embedding_size: Size of the embedding.
    """
    self.id = id
    self.task_id = task_id
    self.observation_type = observation_type
    self.sorted_observation_ids = sorted_observation_ids
    self.embeddings_per_observation = embeddings_per_observation
    self.embedding_size = embedding_size

    super().__init__(len(sorted_observation_ids) * embeddings_per_observation,
                     embedding_size)
    self.counts = np.zeros((len(sorted_observation_ids), 1), dtype=np.int)

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
    idx = self.sorted_observation_ids.index(timestep.observation_id)
    s = idx * self.embeddings_per_observation + self.counts[idx][0]
    self.matrix[s] = embedding
    self.counts[idx][0] += 1

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
    return super().similarity(other)


class Embeddings:
  """State embeddings for multiple tasks."""
  def __init__(self, id: str,
               task_ids: List[TaskId],
               observation_type: ObservationType,
               sorted_observation_ids: List[ObservationId],
               embeddings_per_observation: int,
               embedding_size: int):
    """Creates an `Embeddings` object.

    Args:
      id: An identifier.
      task_ids: Task identifiers for which embeddings are to be recorded.
      observation_type: Type of observation to record. All other types will be ignored.
      sorted_observation_ids: A sorted list of observations to record.
          Observations not included in this list will be dismissed.
      embeddings_per_observation: Number of embeddings per observation id.
      embedding_size: Size of the embedding.
    """
    self.id = id
    self.task_ids = task_ids
    self.observation_type = observation_type
    self.sorted_observation_ids = sorted_observation_ids
    self.embeddings_per_observation = embeddings_per_observation
    self.embedding_size = embedding_size

    self.dataset = {}
    for t in task_ids:
      self.dataset[t] = TaskEmbeddings(
        f'{t}_{id}', t, observation_type,
        sorted_observation_ids,
        embeddings_per_observation, embedding_size)

  def __getitem__(self, task_id):
    return self.dataset[task_id]

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

  def is_compatible(self, other) -> bool:
    if not isinstance(other, Embeddings):
      return False
    if not np.array_equal(self.task_ids, other.task_ids):
      return False
    for t, e in self.dataset:
      if not e.is_compatible(other[t]):
        return False
    return True


def render_similarity(matrix: np.ndarray,
                      labels: List[str],
                      xlabel: str, ylabel: str,
                      specials: Dict[int, str]=None,
                      cmap='RdBu', vmin=-1., vmax=1.,
                      axes=None):
  """Renders a pairwise similarity matrix as a heatmap.

  Args:
    matrix: A 2-dimensional similarity matrix.
    labels: Labels for each row/column.
    xlabel: Label for the x axis.
    ylabel: Label for the y axis.
    specials: Dictionary containing row/column index to mark as keys,
        mapped to a color.

  Returns:
    A tuple containing `matplotlib.figure.Figure` and `matplotlib.axes.Axes`.
  """
  if matrix.ndim < 2 or matrix.shape[0] != matrix.shape[1]:
    raise ValueError(f'Invalid similarity matrix of shape {matrix.shape}')
  if len(labels) != matrix.shape[0]:
    raise ValueError(f'Expected {matrix.shape[0]} labels but got {len(labels)}')

  plt.rcParams.update({'font.size': 22})

  axes_given = True
  if axes is None:
    fig, axes = plt.subplots(figsize=(8, 6))
    axes_given = False

  imshow = axes.imshow(matrix, interpolation=None,
                     cmap=cmap, origin='upper',
                     vmin=vmin, vmax=vmax, aspect='auto')

  # Add ticks.
  ticklabels, indices = np.unique(labels, return_index=True)
  indices = sorted(indices)
  ticklabels = [labels[t] for t in indices]

  dividers = [d - .5 for d in indices]
  gaps = np.append(dividers, len(labels) - .5)
  gaps = (gaps[1:] - gaps[:-1]) / 2.
  ticks = dividers + gaps

  axes.tick_params(axis=u'both', which=u'both', length=0, pad=15)
  axes.set_xlabel(xlabel)
  axes.set_ylabel(ylabel)
  axes.set_yticks(ticks)
  axes.set_xticks(ticks)
  axes.set_yticklabels(ticklabels, ha='center')
  axes.set_xticklabels(ticklabels, ha='center')

  for position in dividers:
    axes.axhline(position, color='k', linestyle='-', linewidth=.2)
    axes.axvline(position, color='k', linestyle='-', linewidth=.2)

  # Add markers if requested.
  if specials is not None:
    for idx, color in specials.items():
      axes.add_artist(matplotlib.patches.Ellipse(
        (-.6, idx), width=.5, height=.5,
        color=color, clip_on=False))
      axes.add_artist(matplotlib.patches.Ellipse(
        (idx, matrix.shape[0] - .35), width=.5, height=.5,
        color=color, clip_on=False))

  if axes_given:
    return

  fig.colorbar(imshow, shrink=.3)
  plt.tight_layout()
  plt.close()

  return fig, axes
