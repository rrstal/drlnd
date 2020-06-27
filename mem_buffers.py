import numpy as np
from collections import namedtuple, deque
import random

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initializes a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size) 
        self.error_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Adds a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self, subset_size=0):    
        """Randomly samples a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    
class PEReplayBuffer:
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
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority"])
        self.seed = random.seed(seed)
        self.e = 0.01
        self.a = 0.7
    
    def add(self, state, action, reward, next_state, done, priority):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, priority)
        self.memory.append(e)
        
    def update_priority(self, idx, errors):
        """Update Priorities in Memory"""

        # add constants to error values
        np_errors = errors.cpu().data.numpy()
        errors_e_a = [(error + self.e)**self.a for error in np_errors.flatten()]

        # include priority in memory
        for (i, err) in zip(idx, errors_e_a): 
            self.memory[i] = self.memory[i]._replace(priority=err)

        
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        # sum up all probabilities
        sum_prio = 0
        for exp in self.memory:
            sum_prio += exp.priority
        
        sampling_probability = [(p.priority / sum_prio) for p in self.memory]
        
        # select indicies by sampling_probability
        idx = np.random.choice(range(self.memory.__len__()), size=self.batch_size, p=sampling_probability, replace=False)
        
        # select experiences in memory by index
        experiences = [self.memory[i] for i in idx]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        priority = torch.from_numpy(np.vstack([e.priority for e in experiences if e is not None])).float().to(device)

        return (idx, states, actions, rewards, next_states, dones, priority)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)