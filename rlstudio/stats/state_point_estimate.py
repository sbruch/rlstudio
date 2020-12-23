from rlstudio.environment import base as env_base
from rlstudio.experiment import base as exp_base
from rlstudio.stats import point_estimate
from rlstudio.typing import EpisodeId, ObservationId, ObservationType

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import warnings


class StatePointEstimate:
  """Records point estimates broken down by state."""
  def __init__(self,
               id: str,
               observation_type: ObservationType,
               observation_ids: List[ObservationId],
               task_ids: List[str],
               horizon: int,
               nrounds: int,
               nruns: int):
    """Creates a `StatePointEstimate` object.

    Args:
      id: A string identifier for this object.
      observation_type: Type of observation to record. All other types will be ignored.
      observation_ids: List of observations to record. Observations not included in this
          list will be dismissed.
      task_ids: List of task identifiers.
      horizon: Number of point estimates per task.
      nrounds: Number of rounds visiting the task series.
      nruns: Number of simulation runs.
    """
    self.id = id
    self.observation_type = observation_type
    self.observation_ids = observation_ids
    self.task_ids = task_ids
    self.horizon = horizon
    self.nrounds = nrounds
    self.nruns = nruns

    # As the main structure, create a dictionary mapping
    # observation IDs to `PointEstimate` objects.
    self.stats = {}
    # Create a temporary buffer that accumulates statistics
    # during an episode, keyed by observation IDs.
    self.buffer = {}
    for observation_id in observation_ids:
      self.stats[observation_id] = point_estimate.PointEstimate(
        id, task_ids, horizon, nrounds, nruns)
      self.buffer[observation_id] = []

  def record(self,
             metadata: exp_base.EvaluationMetadata,
             timestep: env_base.TimeStep,
             estimate: float) -> None:
    """Records an estimate into the buffer."""
    if (self.observation_type != timestep.observation_type or
        timestep.observation_id not in self.stats):
      return
    self.buffer[timestep.observation_id].append(estimate)

  def commit(self, metadata: exp_base.EvaluationMetadata) -> None:
    """Commits values accumulated in the buffer into the main structure."""
    for observation_id, stats in self.stats.items():
      b = self.buffer[observation_id]
      if len(b) > 0:
        stats.record(metadata, np.mean(b))
      self.buffer[observation_id].clear()

  def render_heatmap(self, sorted_observations: List[ObservationId],
                     observation_labels: List[str],
                     xlabel: str, ylabel: str,
                     specials: Dict[ObservationId, str]=None):
    return render_heatmap(self.stats, sorted_observations,
                          observation_labels, xlabel, ylabel, specials)

  def render_compact(self, xlabel: str, ylabel: str, ceiling: float = None, xscale=1):
    """Returns a dict from observation ID to a tuple of matplotlib Figure and Axes."""
    values = {}
    for observation_id, stats in self.stats.items():
      values[observation_id] = stats.render_compact(xlabel, ylabel, ceiling, xscale=xscale)
    return values

  def render_bins(self, bins: Dict[ObservationId, str],
                  episodes: List[EpisodeId],
                  xticklabels: List[str],
                  xlabel: str, ylabel: str):
    return render_bins(self.stats, bins, episodes, xticklabels, xlabel, ylabel)

  def is_compatible(self, other) -> bool:
    if not isinstance(other, StatePointEstimate):
      return False
    if (self.id != other.id or
        self.horizon != other.horizon or
        self.nrounds != other.nrounds or
        self.observation_type != other.observation_type or
        not np.array_equal(self.observation_ids, other.observation_ids) or
        not np.array_equal(self.task_ids, other.task_ids)):
      return False

    for observation, pe in self.stats.items():
      if observation not in other.stats:
        return False
      if not pe.is_compatible(other.stats[observation]):
        return False

    return True


def render_bins(stats: Dict[ObservationId, point_estimate.PointEstimate],
                bins: Dict[ObservationId, str],
                episodes: List[EpisodeId],
                xticklabels: List[str],
                xlabel: str, ylabel: str):
  """Renders statistics for a given set of episodes grouped by the given bins.

  Args:
    stats: Dictionary of `PointEstimate` keyed by observation IDs.
    bins: Dictionary mapping an observation ID to a bin.
    episodes: A list of episodes whose data to render.
    xticklabels: Tick labels on the x-axis.
    xlabel: Label for the x axis.
    ylabel: Label for the y axis.

  Returns:
    A dictionary from plot identifier to a tuple containing
        `matplotlib.figure.Figure` and `matplotlib.axes.Axes`.
  """
  if not np.array_equal(stats.keys(), bins.keys()):
    raise ValueError('Keys in the `stats` and `bins` maps should match: '
                     f'{stats.keys()} vs. {bins.keys()}')
  unique_bins = np.unique([bins[o] for o in bins.keys()])
  if len(xticklabels) != len(unique_bins):
    raise ValueError(f'Insufficient number of xticklabels: got {len(xticklabels)} but '
                     f'expected {len(unique_bins)}')
  if len(episodes) == 0 or len(stats) == 0:
    return {}

  # Validate `PointEstimate`s.
  reference_observation = None
  reference = None
  for observation, pe in stats.items():
    if reference is None:
      reference_observation = observation
      reference = pe
      continue
    if not reference.is_compatible(pe):
      raise ValueError(f'PointEstimates for {reference_observation} and {observation} '
                       'are incompatible')
  for episode in episodes:
    if episode < 0 or episode >= reference.horizon:
      raise ValueError(f'Requested episode {episode} is out of range')

  # Construct a color map.
  cmap = iter(plt.cm.rainbow(np.linspace(0, 1, len(episodes))))
  colors = {}
  for episode in episodes:
    colors[episode] = next(cmap)

  # Generate figures, one per (round, task) tuple.
  figures = {}

  for round in range(reference.nrounds):
    for task_idx, task_id in enumerate(reference.task_ids):
      fig, ax = plt.subplots()
  
      def _plot_episode(episode):
        data: List[np.ndarray] = [np.array([]) for _ in xticklabels]
        for observation, pe in stats.items():
          bin = bins[observation]
          idx = xticklabels.index(bin)
          value = np.squeeze(pe.stats[:, round, task_idx, episode]).reshape((-1))
          data[idx] = np.concatenate([data[idx], value], -1)

        # Compute means and standard deviations for the collected data.
        means = np.zeros((len(data)))
        stds = np.zeros_like(means)
        for idx in range(len(data)):
          with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            means[idx] = np.nanmean(data[idx])
            stds[idx] = np.nanstd(data[idx])

        # Plot.
        x = np.arange(len(xticklabels))
        plt.fill_between(x, means - stds, means + stds,
                         color=colors[episode], alpha=.2)
        plt.plot(x, means, label=f'Time {episode}', c=colors[episode])

      for episode in episodes:
        _plot_episode(episode)

      # Finalize figure.
      ax.set_xticks(np.arange(len(xticklabels)))
      ax.set_xticklabels(xticklabels)
      plt.xlabel(xlabel)
      plt.ylabel(ylabel)
      plt.legend(loc='best')
      plt.tight_layout()
      plt.close()
      figures[f'{task_id}_round {round}'] = (fig, ax)

  return figures


def render_heatmap(stats: Dict[ObservationId, point_estimate.PointEstimate],
                   sorted_observations: List[ObservationId],
                   observation_labels: List[str],
                   xlabel: str, ylabel: str,
                   specials: Dict[ObservationId, str]=None):
  """Renders statistics as a heatmap.

  Args:
    stats: Dictionary of `PointEstimate` keyed by observation IDs.
    sorted_observations: Sorted list of observation IDs.
    observation_labels: Labels for observation IDs in the order they
        appear in `sorted_observations`.
    xlabel: Label for the x axis.
    ylabel: Label for the y axis.
    specials: Dictionary containing observation IDs to mark as keys,
        mapped to a color.

  Returns:
    A dictionary from plot identifier to a tuple containing
        `matplotlib.figure.Figure` and `matplotlib.axes.Axes`.
  """
  if len(observation_labels) != len(sorted_observations):
    raise ValueError('Expected one label per observation')

  if len(stats) == 0:
    return {}

  # Validate `PointEstimate`s.
  reference_observation = None
  reference = None
  for observation, pe in stats.items():
    if reference is None:
      reference_observation = observation
      reference = pe
      continue
    if not reference.is_compatible(pe):
      raise ValueError(f'PointEstimates for {reference_observation} and {observation} '
                       'are incompatible')

  # Find minimum and maximum values.
  vmin = None
  vmax = None
  for _, pe in stats.items():
    _vmin = np.nanmin(pe.stats)
    _vmax = np.nanmax(pe.stats)
    vmin = _vmin if vmin is None else min(vmin, _vmin)
    vmax = _vmax if vmax is None else max(vmax, _vmax)

  # Make vmin and vmax symmetrical around 0.
  if vmin < 0 and vmax > 0:
    if abs(vmin) < abs(vmax):
      vmin = -vmax
    else:
      vmax = -vmin

  # Find ytick positions.
  yticklabels, indices = np.unique(observation_labels, return_index=True)
  indices = sorted(indices)
  yticklabels = [observation_labels[t] for t in indices]

  ydividers = [d - .5 for d in indices]
  gaps = np.append(ydividers, len(observation_labels) - .5)
  gaps = (gaps[1:] - gaps[:-1]) / 2.
  yticks = ydividers + gaps

  # Plot one figure per task.
  figures = {}
  for task_idx, task_id in enumerate(reference.task_ids):
    image = np.zeros((len(sorted_observations), reference.nrounds * reference.horizon))
    for observation, pe in stats.items():
      row = sorted_observations.index(observation)
      with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        image[row] = np.nanmean(pe.stats[:, :, task_idx, :],
                                axis=0).reshape((image.shape[-1]))

    fig, ax = plt.subplots(figsize=(8, 6))
    imshow = ax.imshow(image, interpolation='none',
                       cmap='RdBu', origin='upper',
                       vmin=vmin, vmax=vmax)

    # Labels, title, and ticks.
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis=u'x', which=u'major', bottom=False,
                   top=False, labelbottom=False)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, ha='center')
    ax.tick_params(axis=u'y', which=u'both', length=0, pad=15)
    for position in ydividers:
      ax.axhline(position, color='k', linestyle='-', linewidth=.5)
    
    # Add dividers to the x axis.
    if reference.nrounds > 1:
      minor_ticks = []
      x = 0
      for _ in range(reference.nrounds):
        minor_ticks.append(x + reference.horizon / 2)
        x += reference.horizon
        ax.axvline(x - .5, color='k', linestyle='-', linewidth=.5)

      ax.set_xticks(minor_ticks, minor=True)
      ax.set_xticklabels([f'Round {i+1}' for i, _ in enumerate(minor_ticks)],
                         minor=True, ha='center')
      ax.tick_params(axis=u'x', which=u'minor', pad=25)

    # Update aspect ratio.
    _, xmax = ax.get_xlim()
    ratio = xmax / len(sorted_observations)
    if ratio < 1:
      ratio = 1
    ax.set_aspect(ratio)

    # Add markers if requested.
    if specials is not None:
      x = 0
      for _ in range(reference.nrounds):
        for observation, color in specials.items():
          y = sorted_observations.index(observation)
          ax.add_artist(matplotlib.patches.Ellipse(
            (x-.6, y), width=.3 * ratio, height=.3,
            color=color, clip_on=False))
        x += reference.horizon

    fig.colorbar(imshow, shrink=.3)
    plt.tight_layout()
    plt.close()
    figures[task_id] = (fig, ax)

  return figures

def unify(points: List[StatePointEstimate]) -> StatePointEstimate:
  """A mechanism to aggregate multiple `StatePointEstimate` objects into one unified object."""
  if len(points) == 0:
    return None

  # Validate and compute total number of runs.
  nruns = points[0].nruns
  for point in points[1:]:
    if not points[0].is_compatible(point):
      raise ValueError('Incompatible StatePointEstimate objects')
    nruns += point.nruns

  # Unify.
  unified = StatePointEstimate(
    id=points[0].id,
    observation_type=points[0].observation_type,
    observation_ids=points[0].observation_ids,
    task_ids=points[0].task_ids,
    horizon=points[0].horizon,
    nrounds=points[0].nrounds,
    nruns=nruns)

  runs = 0
  for point in points:
    for o, pe in point.stats.items():
      unified.stats[o].stats[runs:runs+point.nruns] = pe.stats
    runs += point.nruns

  return unified
