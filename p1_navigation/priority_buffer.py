from collections import namedtuple, deque
import torch
import random
from device import device
import numpy as np

class PriorityBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
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
                                     field_names=["state", "action", "reward", "next_state", "done", "priority"])
        self.seed = random.seed(seed)
        self.e = 1e-3
        self.unform_sample_factor = 0.5  # 0 means completely uniform

    def add(self, state, action, reward, next_state, done, priority):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, priority)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        #         prios = np.asarray([experience.priority for experience in self.memory])
        #         prios+=self.e
        #         prios = prios**self.unform_sample_factor
        #         prios_sum=np.sum(prios)

        #         probs = prios/prios_sum
        experiences = random.sample(self.memory, k=self.batch_size)
        #         experiences = random.choices(self.memory, weights= probs, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        prios = np.array([e.priority for e in experiences if e is not None])
        #         prios=prios**self.unform_sample_factor/prios_sum
        probs = torch.from_numpy(np.vstack(prios)).float().to(device)

        return (states, actions, rewards, next_states, dones, probs)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
