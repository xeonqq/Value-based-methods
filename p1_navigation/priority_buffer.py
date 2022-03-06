import random
from collections import deque

import numpy as np
import torch
from recordtype import recordtype

from device import device


class PriorityBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, min_non_zero_error=1e-4,
                 uniform_sample_factor_alpha=0.9):
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
        self.experience = recordtype("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done", "abs_td_error", "importance_sampling_weight"])
        self.seed = random.seed(seed)
        self.min_non_zero_error = min_non_zero_error
        self.uniform_sample_factor = uniform_sample_factor_alpha  # 0 means completely uniform
        self.max_td_error = self.min_non_zero_error
        self.sampled_indexes = []
        self.importance_sampling_weights=[]
        self.max_w = 1;

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, self.max_td_error, 0)
        self.memory.append(e)

    def _select_sampled_experiences(self):
        return [self.memory[i] for i in self.sampled_indexes]

    def _update_importance_sampling_weights(self, probs, beta):
        self.importance_sampling_weights = (probs*len(self.memory))**(-beta)
        self.max_w = np.max(self.importance_sampling_weights)

    def sample(self, beta=1):
        """Randomly sample a batch of experiences from memory."""
        probs = self._abs_td_errors_to_probs()
        self.sampled_indexes = np.random.choice(len(self.memory), size=self.batch_size, replace=False, p=probs)
        sampled_experiences = self._select_sampled_experiences()

        self._update_importance_sampling_weights(probs, beta)

        states = torch.from_numpy(np.vstack([e.state for e in sampled_experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in sampled_experiences if e is not None])).long().to(
            device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in sampled_experiences if e is not None])).float().to(
            device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in sampled_experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in sampled_experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        sampled_probs = probs[self.sampled_indexes]
        sampled_probs = torch.from_numpy(np.vstack(sampled_probs)).float().to(device)
        sampling_weights = self.importance_sampling_weights[self.sampled_indexes]/self.max_w
        sampling_weights = torch.from_numpy(np.vstack(sampling_weights)).float().to(device)

        return (states, actions, rewards, next_states, dones, sampled_probs, sampling_weights)

    def _abs_td_errors_to_probs(self):
        abs_td_errors = np.asarray([experience.abs_td_error for experience in self.memory])
        td_errors_factored = abs_td_errors ** self.uniform_sample_factor
        probs = td_errors_factored / np.sum(td_errors_factored)
        return probs

    def update_td_errors(self, td_errors):
        experiences = self._select_sampled_experiences()
        new_abs_td_errors = np.abs(np.squeeze(td_errors)) + self.min_non_zero_error
        for experience, abs_td_error in zip(experiences, new_abs_td_errors):
            experience.abs_td_error = abs_td_error
        self.max_td_error = max(np.max(new_abs_td_errors), self.max_td_error)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
