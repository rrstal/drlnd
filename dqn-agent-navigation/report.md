
**Goal:**



The project aims at familiarizing with DQN Learning for a simple navigation task. The agent is in a 3D world (practically in a 2D since the possible movements are restricted to a 2D plane) surrounded by blue and yellow bananas. All the agent has to learn is to collect as many yellow bananas as possible while avoiding the blue ones. In practice it means that the agent finds a yellow banana next to it and moves in its direction. There are a few constraints to the task: reward of at least 13 has to be achieved in as short time as possible.
	
**Learning Algorithm:**


The starting point to solve the task has been the basic version of DQN. It worked pretty well with the given settings utilized in the lesson. As a matter of practice I decided to implement follow-ups of the standard DQN: Double DQN and Prioritized Experience Buffer. Results of all three agents are listed below. Metrics used to track performance have been the max average score and the time it took to reach the reward threshold of 13.
There has been no real performance difference for all the 3 implemented methods in stability, converging time or value. As a one reason, the given task is relatively simple. Obviously, tweaking hyperparameters and possibly the DQN NN architecture would improve the results. Further improvement ideas are listed below.

**Hyperparameters:**

```
	BUFFER_SIZE = int(1e5)  # replay buffer size
	BATCH_SIZE = 64         # minibatch size
	GAMMA = 0.99            # discount factor
	TAU = 1e-3              # for soft update of target parameters
	LR = 5e-4               # learning rate 
	UPDATE_EVERY = 4        # how often to update the network
```
	
**DQN NN Architecture**
* hidden layer 1 = 16 units with relu activation
* hidden layer 2 = 16 units with relu activation
* hidden layer 3 = 16 units with relu activation
* loss = mean squared error


**Ideas for Future Work:**
* implement more advanced versions of DQN such as dueling or rainbow. 
* learn directly from raw pixel input. Convolutional NN architecture would be the way to go. 
	
	
**Results:**
	
* basic DQN:
```
eps_start=1.0, eps_end=0.01, eps_decay=0.995
Environment solved in 524 episodes!	Average Score: 13.10
```
* Double DQN:
```
eps_start=1.0, eps_end=0.01, eps_decay=0.995
Environment solved in 502 episodes!	Average Score: 13.01
```
* basic DQN with Prioritized Experience Replay:
```
eps_start=1.0, eps_end=0.01, eps_decay=0.995
Environment solved in 551 episodes!	Average Score: 13.01
```


Plots show scores and smoothed scores, over all episodes, for each of the implemented agents.

![scores](https://github.com/rrstal/drlnd-dqn-agent-navigation/blob/master/media/results.PNG)

![smoothed_scores](https://github.com/rrstal/drlnd-dqn-agent-navigation/blob/master/media/smoothed_results.PNG)