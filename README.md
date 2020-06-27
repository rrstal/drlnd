[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Agent in action"

# Navigation

## Description

The goal of this project is to train an agent to navigate and collect bananas in a large, square world. 


![Agent in action][image1]

## Problem Statement

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided 
for collecting a blue banana. Thus, the goal of the agent is to collect 
as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions, and contains the agent's velocity, along
with ray-based perception of objects around the agent's forward
direction. Given this information, the agent has to learn how to best select 
actions. 
Four discrete actions are available, corresponding to: 
- `0` - move forward
- `1` - move backward
- `2` - turn left
- `3` - turn right
The task is episodic, and in order to solve the environment, the 
agent must get an average score of +13 over 100 consecutive episodes.

## Files

- `agents.py`: implementation of the agents used in the environment
- `models.py`: implementation of the Q-Network architectures used as the function approximator by the agent
- `dqn.pth`: saved model weights for the original DQN model
- `ddqn.pth`: saved model weights for the Double DQN model
- `ddqn_priority.pth`: saved model weights for the Dueling Double DQN Priority Experience Replay model
- `Navigation.ipynb`: notebook containing the solution, **entry point**.

### Weights

- **DQN**: to run the basic DQN algorithm, checkpoint `dqn.pth` contains the trained model
- **Double DQN**: to run the Double DQN algorithm, use the checkpoint `ddqn.pth` for loading the trained model.
- **PER DQN**: to run the Prioritised Experience Replay algorithm, use the checkpoint `ddqn_priority.pth` for loading the trained model.


## Dependencies
Code has been implemented in python 3,
the other dependencies are listed in the `requirements.txt` file. Install them with the following command:

```
pip install requirements.txt
```

Furthermore, you the Unity ML Agents environment from one of the links below. Select the one matching your operating system:
- Linux : [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- MAC OSX : [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows : [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)


Place the file in the root project location, decompress the file and then update the path argument for creating the environment in `Navigation.ipynb`:

Example for Mac OS:
```python
env = UnityEnvironment(file_name="Banana.app")
```

## Execution
Run the cells in the notebook `Navigation.ipynb` to train an agent.


## Implementations
Repository contains implementation of the further enhancements to the original DQN agent:

- Double DQN [[Paper](https://arxiv.org/abs/1509.06461)]
- Prioritized Experience Replay [[Paper](https://arxiv.org/abs/1511.05952)]
