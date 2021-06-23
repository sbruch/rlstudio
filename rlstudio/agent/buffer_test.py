from rlstudio.agent import buffer as blib
from rlstudio.environment import base

from absl.testing import absltest
from dm_env import specs
import numpy as np


class BufferTest(absltest.TestCase):
  def test_buffer(self):
    # Initialize the buffer.
    max_trajectory_length = 10
    observation_shape = (3, 3)
    buffer = blib.Buffer(
      observation_spec=specs.Array(observation_shape, dtype=np.float32),
      action_spec=specs.Array((), dtype=np.int32),
      max_trajectory_length=max_trajectory_length)
    dummy_step = base.transition(
      action=0., reward=0.,
      observation=np.zeros(observation_shape))

    # Fill the buffer.
    for _ in range(max_trajectory_length):
      buffer.append(dummy_step, dummy_step)

    self.assertTrue(buffer.full())

    # Any further appends should fail.
    with self.assertRaises(ValueError):
      buffer.append(dummy_step, dummy_step)

    # Drain the buffer.
    trajectory = buffer.drain()

    self.assertLen(trajectory.observations, max_trajectory_length + 1)
    self.assertLen(trajectory.actions, max_trajectory_length)
    self.assertLen(trajectory.rewards, max_trajectory_length)
    self.assertLen(trajectory.discounts, max_trajectory_length)

    self.assertTrue(buffer.empty())
    # Draining an empty buffer should fail.
    with self.assertRaises(ValueError):
      buffer.drain()

    # Add an entry and immediately drain the buffer.
    buffer.append(dummy_step, dummy_step)
    trajectory = buffer.drain()

    self.assertLen(trajectory.observations, 2)
    self.assertLen(trajectory.actions, 1)
    self.assertLen(trajectory.rewards, 1)
    self.assertLen(trajectory.discounts, 1)


if __name__ == '__main__':
  absltest.main()
