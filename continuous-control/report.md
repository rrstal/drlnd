
**Goal:**

The project aims at demonstrating how to use policy-based approaches to Reinforcement Learning in order to learn the optimal policy. The provided Unity environment 'Reacher' serves as the simulation where each of the agents is a double-jointed arm that can move to target locations. A reward of +0.1 is provided for each step that the agent's hands is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.
	
**Learning Algorithm:**

The solution contains implementation along the idea of Deep Deterministic Policy Gradients: [[Paper](https://arxiv.org/abs/1509.02971)]

DDPG is an Actor-Critic method that uses value function and direct policy approximation at the same time.
There are two types of neural networks:
- Actor network - transforms state to action values.
- Critic network - transforms state and action values to a quality measure of this state (Q(s, a))

The idea of usage of both the value function (Critic network) & policy approximation (Actor network) mitigates the problems of each of the approaches alone:
- policy based methods - suffering from a high variance
- value function methods - suffering from a high bias

DDPG can be thought as a continuous version of DQN, with a continuous action space.

Other than that, the implementation is similar to the one of DQN. With the following components:

**Experience Replay** 

A buffer with experience tuples (s, a, r, s'): (state, action, reward, next_state)

**Q-targets fixing**

2 neural networks (NN): local and target.
To improve training stability and make diverging or oscillations less likely, the weights of the target NN are fixed every certain amount of steps and thus decouple the local NN from target NN. After those steps, they are updated and the fixing process repeats.

Practically there are 4 neural networks altogether:
- Critic target NN
- Critic local NN (for execution) 
- Actor target NN 
- Actor local NN (for execution)


**Hyperparameters:**

```
	BUFFER_SIZE = int(1e5)  # replay buffer size
	BATCH_SIZE = 1024       # minibatch size
	GAMMA = 0.99            # discount factor
	TAU = 1e-3              # for soft update of target parameters
	LR_ACTOR = 1e-4         # learning rate actor
	LR_CRITIC = 3e-4		# learning rate critic
	WEIGHT_DECAY = 0.0001	# L2 weight decay
	LEAKINESS = 0.01		# leaky ReLU param
```
	
**Actor NN Architecture**
* hidden layer 1 = 256 units with relu activation
* hidden layer 2 = 128 units with relu activation
* loss = mean squared error

**Critic NN Architecture**
* hidden layer 1 = 256 units with relu activation
* hidden layer 2 = 128 units with relu activation
* hidden layer 3 = 128 units with relu activation
* loss = mean squared error

**Ideas for Future Work:**
* continue to tune more the hyperparameters
* refine neural network models: actor, critic
* try out other algorithms, especially in the case of multiple agents: **PPO**, **A3C**
* **Prioritized Experience Replay**: the idea behind is that we want to focus during training on the experiences which convey lot of learning potential, whose actions were far from good. This means, that their TD error was respectively higher. It maps to higher priority of sampling them from the buffer. Additionally, to avoid situation where experience would hold a probability 0 and thus be never seen again during training, we need to add a small epsilon value to each probability.
	
**Results:**

```
Episode: 318	Average Score: 30.07	Current Score: 34.09
Environment solved in 218 episodes!	Average Score: 30.07
```


Plot shows scores per episode, being an average over all the agents, together with the threshold. Required average over the last 100 episodes for all the agents was set to be 30.

![scores](https://github.com/rrstal/drlnd/blob/master/continuous-control/media/result.png)