import random
from collections import namedtuple, deque

import numpy as np
import torch

from device import device


class PriorityBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, e_for_non_zero=1e-3, uniform_sample_factor=0.5):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done", "abs_td_error"])
        self.seed = random.seed(seed)
        self.e = e_for_non_zero
        self.uniform_sample_factor = uniform_sample_factor  # 0 means completely uniform

    def add(self, state, action, reward, next_state, done, abs_td_error):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, abs_td_error)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        probs = self.abs_td_errors_to_probs()
        experience_inds = np.random.choice(len(self.memory), size=self.batch_size, replace=False, p=probs)
        experiences = [self.memory[i] for i in experience_inds]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        sampled_probs = probs[experience_inds]
        sampled_probs = torch.from_numpy(np.vstack(sampled_probs)).float().to(device)

        return (states, actions, rewards, next_states, dones, sampled_probs)

    def abs_td_errors_to_probs(self):
        abs_td_errors = np.asarray([experience.abs_td_error for experience in self.memory])
        abs_td_errors += self.e
        td_errors_factored = abs_td_errors ** self.uniform_sample_factor
        probs = td_errors_factored / np.sum(td_errors_factored)
        return probs

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
