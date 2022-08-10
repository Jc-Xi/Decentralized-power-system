from collections import deque, namedtuple
import warnings
import random

import numpy as np


def sample_batch_indexes(memory_size, batch_size):
    if memory_size >= batch_size:
        r = range(memory_size)
        batch_idxs = random.sample(r, batch_size)
    else:
        # Not enough data. Help ourselves with sampling from the range, but the same index
        # can occur multiple times. This is not good and should be avoided by picking a
        # large enough warm-up phase.
        warnings.warn('Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
        batch_idxs = np.random.random_integers(0, memory_size-1, size=batch_size)
    assert len(batch_idxs) == batch_size
    return batch_idxs
class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v
class Memory(object):
    def __init__(self,limit):
        self.limit = limit
        self.actions = RingBuffer(limit)
        self.observations = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.next_observations = RingBuffer(limit)
    def sample_and_split(self,batch_size,batch_idxs = None):
        if batch_idxs is None:
            batch_idxs = sample_batch_indexes(self.nb_entries,batch_size)
        observation_batch = np.array([self.observations[idx] for idx in batch_idxs]).reshape(batch_size,-1)
        action_batch = np.array([self.actions[idx] for idx in batch_idxs]).reshape(batch_size,-1)
        reward_batch = np.array([self.rewards[idx] for idx in batch_idxs]).reshape(batch_size,-1)
        next_obs_batch = np.array([self.next_observations[idx] for idx in batch_idxs]).reshape(batch_size,-1)

        return observation_batch,action_batch,reward_batch,next_obs_batch
    def append(self,observation,action,reward,next_observation):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_observation)
    @property
    def nb_entries(self):
        return len(self.observations)