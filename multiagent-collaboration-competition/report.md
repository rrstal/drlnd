**Goal:**

The project aims at demonstrating how to use multi agent off-policy approaches to Reinforcement Learning in order to learn an environment of competitive or collaborative agents. In this specific project, the provided environment 'Tennis' serves as the simulation where each of the two agents controls its own racket to bounce a ball over a net. Once agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observatino. Two continuous actions are available, corresponding to movement towards or away from the net, and jumping.
    
**Learning Algorithm:**

The solution contains implementation inspired by the idea of Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments [[Paper](https://arxiv.org/abs/1706.02275)]

Since MADDPG is DDPG extended to multiple agents, a brief overview of the DDPG comes first as follows.

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

In multi agent environment, DDPG is used to control individual agents, however the critic networks are shared between agents. Sharing and training is realised via sampling on a shared memory replay buffer. Each agent samples its own experience tuple. Similarly, action noise is added for each agent individually.

**Hyperparameters:**
```
    buffer_size = int(1e4)  # replay buffer size
    batch_size = 256        # minibatch size
    gamma = 0.99            # discount factor
    tau = 1e-3              # for soft update of target parameters
    lr_actor = 1e-4         # learning rate actor
    lr_critic = 3e-4        # learning rate critic
    noise_start = 0.5       # initial value of the action noise
```
    
**Actor NN Architecture**
* hidden layer 1 = 256 units with relu activation
* batchnorm layer 1
* hidden layer 2 = 256 units with relu activation
* loss = mean squared error

**Critic NN Architecture**
* hidden layer 1 = 256 units with relu activation
* batchnorm layer 1
 hidden layer 2 = 256 units with relu activation
* loss = mean squared error

**Ideas for Future Work:**


One of the main challenges was hyperparameter tuning. Especially it turned out that the proper amount of noise is of great importance. Since the noise is added to actions, too few of it would make the agent strongly dependent on the initialization and fragile to get stuck in local optima. Too much noise would cause agent to take extreme actions, which might not be desired once the agent is on the good track of learning. Thus, one idea would be to introduce a more adaptive way of deciding about the amount of noise. For example dependent on the number of episodes or gradient of the rewards.

It can be observed that the rewards oscillated a lot. This poses a question of robustness of the agent. Introducing some changes to the scenario itself (e.g. size of the court, size or weight of the ball that would influence its dynamics, and such) and conducting some additional trainings could give a bit sense about that.

One more immediate improvement idea would be to introduce more sophisticated sampling mechanism for the shared memory buffer, for example Prioritized Experience Replay. The idea behind is that one wants to focus during training on the experiences which convey lot of learning potential, whose actions were far from good. This means, that their TD error was respectively higher. It maps to higher priority of sampling them from the buffer. Additionally, to avoid situation where experience would hold a probability 0 and thus be never seen again during training, one needs to add a small epsilon value to each probability.
    
**Results:**

```
Environment solved, the agents scored on average above the required threshold of 0.50
Episode:5108, Current Average Score:0.50
```

Plot shows scores per episode, being max from both the agents, together with the threshold. Required average over the last 100 episodes was set to be 0.5.

![scores](https://github.com/rrstal/drlnd/blob/master/multiagent-collaboration-competition/media/maddpg.PNG)