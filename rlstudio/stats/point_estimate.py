from rlstudio.experiment import base as exp_base
from rlstudio.typing import TaskId

import matplotlib.pyplot as plt
import numpy as np
from typing import List
import warnings


class PointEstimate:
  """Records point estimates such as returns."""
  def __init__(self,
               id: str,
               task_ids: List[TaskId],
               horizon: int,
               nrounds: int,
               nruns: int):
    """Creates a PointEstimate object.
    
    Args:
      id: A string identifier for this object.
      task_ids: List of task identifiers.
      horizon: Number of point estimates per task.
      nrounds: Number of rounds visiting the task series.
      nruns: Number of simulation runs.
    """
    self.id = id
    self.task_ids = task_ids
    self.horizon = horizon
    self.nrounds = nrounds
    self.nruns = nruns

    self.stats = np.zeros((nruns, nrounds, len(task_ids), horizon))
    self.stats[:] = np.nan

  def record(self, metadata: exp_base.EvaluationMetadata, estimate: float) -> None:
    """Records an estimate."""
    r = metadata.round_id
    t = self.task_ids.index(metadata.task_id)
    h = metadata.time % self.horizon
    run = metadata.run_id % self.nruns
    self.stats[run, r, t, h] = estimate

  def render_sequential(self, xlabel: str, ylabel: str,
                        xticks=None, ceiling: float = None, xscale=1):
    """Renders the accumulated statistics one task at a time.

    Args:
      xlabel: Label for the x axis.
      ylabel: Label for the y axis.
      xticks: Optional xticks. Computed automatically if not given.
      ceiling: Optional maximum value achievable. A dashed horizontal line
          is plotted to highlight this value.
      xscale: Scales ticks on x axis.

    Returns:
      A tuple containing `matplotlib.figure.Figure` and `matplotlib.axes.Axes`.
    """
    # Construct a color map.
    unique_task_ids = sorted(np.unique(self.task_ids))
    cmap = iter(plt.cm.rainbow(np.linspace(0, 1, len(unique_task_ids))))
    colors = {}
    for task_id in unique_task_ids:
      colors[task_id] = next(cmap)

    # Plot.
    fig, ax = plt.subplots()

    current_x = 0
    for round_id in range(self.nrounds):
      for task_idx, task_id in enumerate(self.task_ids):
        data = np.squeeze(self.stats[:, round_id, task_idx, :])
        x = np.arange(current_x, current_x + data.shape[-1]) * xscale

        if data.ndim == 2:
          with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            y = np.nanmean(data, axis=0)
            err = np.nanstd(data, axis=0)
          plt.fill_between(x, y - err, y + err,
                           color=colors[task_id], alpha=.2)
        else:
          y = data

        plt.plot(x, y, c=colors[task_id], label=task_id,
                 marker='o' if len(y) == 1 else None)
        current_x += data.shape[-1]

    # Remove duplicates from entries in the legend.
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels, loc='best')

    if ceiling is not None:
      ax.axhline(ceiling, color='k', linestyle=':', linewidth=.75)

    # Add labels.
    if xticks is not None:
      plt.xticks(xticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.close()

    return fig, ax

  def render_compact(self, xlabel: str, ylabel: str,
                     xticks=None, ceiling: float = None, xscale=1):
    """Renders the accumulated statistics in a compact plot.

    Args:
      xlabel: Label for the x axis.
      ylabel: Label for the y axis.
      xticks: Optional xticks. Computed automatically if not given.
      ceiling: Optional maximum value achievable. A dashed horizontal line
          is plotted to highlight this value.
      xscale: Scales ticks on x axis.

    Returns:
      A tuple containing `matplotlib.figure.Figure` and `matplotlib.axes.Axes`.
    """
    # Construct a color map.
    unique_task_ids = sorted(np.unique(self.task_ids))
    cmap = iter(plt.cm.rainbow(np.linspace(0, 1, len(unique_task_ids))))
    colors = {}
    for task_id in unique_task_ids:
      colors[task_id] = next(cmap)

    # Plot.
    fig, ax = plt.subplots()

    for task_idx, task_id in enumerate(self.task_ids):
      data = self.stats[:, :, task_idx, :].reshape((self.nruns, -1))
      data = np.squeeze(data)
      x = np.arange(data.shape[-1]) * xscale

      if data.ndim == 2:
        with warnings.catch_warnings():
          warnings.simplefilter("ignore", category=RuntimeWarning)
          y = np.nanmean(data, axis=0)
          err = np.nanstd(data, axis=0)
        plt.fill_between(x, y - err, y + err,
                         color=colors[task_id], alpha=.2)
      else:
        y = data

      plt.plot(x, y, c=colors[task_id], label=task_id,
               marker='o' if len(y) == 1 else None)

    plt.legend(loc='best')

    if ceiling is not None:
      ax.axhline(ceiling, color='k', linestyle=':', linewidth=.75)

    # Add labels.
    if xticks is not None:
      plt.xticks(xticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.close()

    return fig, ax

  def is_compatible(self, other) -> bool:
    if not isinstance(other, PointEstimate):
      return False
    if self.id != other.id or self.horizon != other.horizon or self.nrounds != other.nrounds:
      return False
    if not np.array_equal(self.task_ids, other.task_ids):
      return False
    return True


def unify(points: List[PointEstimate]) -> PointEstimate:
  """A mechanism to aggregate multiple `PointEstimate` objects into one unified object."""
  if len(points) == 0:
    return None

  # Validate and compute total number of runs.
  nruns = points[0].nruns
  for point in points[1:]:
    if not points[0].is_compatible(point):
      raise ValueError('Incompatible PointEstimate objects')
    nruns += point.nruns

  # Unify.
  unified = PointEstimate(
    id=points[0].id,
    task_ids=points[0].task_ids,
    horizon=points[0].horizon,
    nrounds=points[0].nrounds,
    nruns=nruns)

  runs = 0
  for point in points:
    unified.stats[runs:runs+point.nruns] = point.stats
    runs += point.nruns

  return unified
