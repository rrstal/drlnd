import numpy as np
import random

import torch

from neural_models import ActorCriticWrapper
from ddpg import DDPGAgent
from mem_buffers import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiAgentWrapper():
    """Wrapper over Agents that provides functionality of a single agent and enables shared memory buffer."""
        
    def __init__(self, 
                 action_size=2, 
                 seed=42, 
                 n_agents=2,
                 state_size=24,
                 buffer_size=10000,
                 batch_size=256,
                 gamma=0.99,
                 noise_start=1.0,
                 noise_decay=1.0):
        """Initialize parameters and build multi agent wrapper.
        Params
        ======
            action_size (int): dimension of each action
            seed (int): Random seed
            n_agents (int): number of distinct agents
            state_size (int): size of the state space
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            noise_start (float): initial noise weighting factor
            noise_decay (float): noise decay rate
        """
        self.action_size = action_size
        self.seed = seed
        self.n_agents = n_agents
        self.state_size = state_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.noise_weight = noise_start
        self.noise_decay = noise_decay
        
        self.enable_noise = True

        # instantiate agents with respective actor and critic
        models = [ActorCriticWrapper(num_agents=self.n_agents) for _ in range(self.n_agents)]
        self.agents = [DDPGAgent(i, models[i]) for i in range(self.n_agents)]
        
        # instantiate shared replay buffer
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.seed)

    def step(self, states, actions, rewards, next_states, dones):
        """Move to the next state, collect experience into buffer, sample experience and learn."""
        
        states = states.reshape(1, -1)
        next_states = next_states.reshape(1, -1)
        self.memory.add(states, actions, rewards, next_states, dones)

        # for each agent, sample experiences from the shared buffer and learn
        if len(self.memory) > self.batch_size:
            experiences = [self.memory.sample() for _ in range(self.n_agents)]
            self.learn(experiences, self.gamma)

    def act(self, states, add_noise=True):
        """Read state of each agent and calculate its action."""
      
        actions = []
        for agent, state in zip(self.agents, states):
            action = agent.act(state, noise_weight=self.noise_weight, add_noise=self.enable_noise)
            actions.append(action)
            self.noise_weight *= self.noise_decay
        return np.array(actions).reshape(1, -1) # flatten

    def learn(self, experiences, gamma):
        """Calculate next actions and perform learning for each agent."""
        
        next_actions = []
        actions = []
        for i, agent in enumerate(self.agents):
            
            # collect current agent's ID, current and next states
            states, _, _, next_states, _ = experiences[i]
            agent_id = torch.tensor([i]).to(device)
            
            # extract the state of an agent and get action provided by actor 
            state = states.reshape(-1, self.n_agents, self.state_size).index_select(1, agent_id).squeeze(1)
            action = agent.actor_local(state)
            actions.append(action)
            
            # extract the next state of an agent and get action provided by target actor
            next_state = next_states.reshape(-1, self.n_agents, self.state_size).index_select(1, agent_id).squeeze(1)
            next_action = agent.actor_target(next_state)
            next_actions.append(next_action)
                       
        # perform learning for each agent, from its own sampled experience
        for i, agent in enumerate(self.agents):
            agent.learn(i, experiences[i], gamma, next_actions, actions)