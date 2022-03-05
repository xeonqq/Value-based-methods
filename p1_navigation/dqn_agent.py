import numpy as np
import random
from device import device
from replay_buffer import ReplayBuffer
from priority_buffer import PriorityBuffer

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, use_priority_buffer=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.use_priority_buffer=use_priority_buffer
        # Replay memory
        if self.use_priority_buffer:
            self.memory = PriorityBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def td_error(self, state, action, reward,next_state, done):
        next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)

        target_action_values_next = self.qnetwork_target(next_state_tensor)
        target_action_values_next = target_action_values_next.cpu().data.numpy()
        Q_targets_next = np.max(target_action_values_next)
        # Compute Q targets for current states
        Q_targets = reward + (GAMMA * Q_targets_next * (1 - done))
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()
        Q_expected = action_values.cpu().data.numpy()[0][action]
        return Q_targets-Q_expected

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        # error = self.td_error(state, action, reward, next_state, done)
        if self.use_priority_buffer:
            self.memory.add(state, action, reward, next_state, done)
        else:
            self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
#         return state, action, reward, next_state, done,priority

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        if self.use_priority_buffer:
            states, actions, rewards, next_states, dones, probs = experiences
        else:
            states, actions, rewards, next_states, dones = experiences
        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"
        # Get max predicted Q values (for next states) from target model
        # Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        self.qnetwork_local.eval()
        with torch.no_grad():
            argmax_actions_from_local_Q = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            Q_targets_values_next = self.qnetwork_target(next_states).detach()
            Q_targets_next = Q_targets_values_next.gather(1, argmax_actions_from_local_Q)
        self.qnetwork_local.train()

        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
#         sampling_weights = (1/BUFFER_SIZE*1/probs)**b
        
#         loss = F.mse_loss(Q_expected*sampling_weights, Q_targets*sampling_weights)
        
        loss = F.mse_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step() 
        


        # ------------------- update target network ------------------- #
        
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
