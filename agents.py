import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
from mem_buffers import PEReplayBuffer, ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, name=None, ddqn=False, priority=False):
        """Initializes an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            ddqn (bool): flag for double dqn
            priority (bool): flag for priority replay buffer
        """
        assert name, "Name of the agent has to be defined but is None."
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.ddqn = ddqn
        self.priority = priority
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        if self.priority:
            self.memory = PEReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        if self.ddqn:
            self.t_step = 0
        else:
            self.t_step = 10
            
        if self.priority:
            self.t_step = 0
            self.beta = 0.5
    
    def step(self, state, action, reward, next_state, done):
        """Saves experiences in memory and fetches them in each step.
        
        Params
        ======
            state (array_like): current state
            action (array_like): current actions
            reward (array_like): current rewards
            next_state (array_like): next state
            done (bool): flag for end of the episode
        """

        if self.priority:
            # initial priority error
            error = 1.0
            # Save experience in replay memory with priority
            self.memory.add(state, action, reward, next_state, done, error)
        else:        
            # Save experience in replay memory
            self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # Given enough available samples in memory buffer, fetch random subset
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                
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
        """Updates value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        
        if self.priority:
            idx, states, actions, rewards, next_states, dones, priority = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        if self.ddqn:
            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).detach()
            best_actions = self.qnetwork_local(next_states).detach().max(1)[1]
            # select action from qnetwork_target based on best_actions(qnetwork_local)
            Q_targets_next = Q_targets_next[np.arange(BATCH_SIZE), best_actions].unsqueeze(1)
        else:
            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions) 
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        if self.priority:
            # get Error for Prioritized Experience Replay & Update priority
            error = abs(Q_expected - Q_targets)
            self.memory.update_priority(idx, error)
            
            # bias annealing
            w_i = (self.memory.__len__() * priority)**-self.beta
            # normalize weights
            w_j = w_i/max(w_i)
            # Compute loss (mse)
            loss = (Q_expected - Q_targets)*w_j
            loss = sum(loss**2)/len(loss)
        else:
            # Compute loss
            loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft updates model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)