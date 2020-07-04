[//]: # (Image References)

# Continuous Control

## Description

The goal of this project is to train multiple double-jointed arm agents to follow, each one of them, a target location.

<p align="center">
    <img src="media/reacher_multiagent.gif" width=50% height=50%>
</p>

## Problem Statement

A reward of +0.1 is provided for each step that the agent's hands is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic, with 1000 timesteps per episode. In order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.


## Files

- `ddpg_agent.py`: implementation of the agents used in the environment
- `models.py`: implementation of the actor and the critic classes
- `checkpoint_critic.pth`: saved model weights for the Critic
- `checkpoint_actor.pth`: saved model weights for the Actor
- `Continuous_Control.ipynb`: notebook containing the solution, **entry point**


## Dependencies
Code has been implemented in python 3,
the other dependencies are listed in the `requirements.txt` file. Install them with the following command:

```
pip install requirements.txt
```

Furthermore, you need to download the Unity ML Agents environment from one of the links below. Select the one matching your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Place the file in the root project location, decompress the file and then update the path argument for creating the environment in `Continuous_Control.ipynb`:

Example for Mac OS:
```python
env = UnityEnvironment(file_name="Reacher.app")
```

## Execution
Run the cells in the notebook `Continuous_Control.ipynb` to train the group of agents.


## Implementations
Repository contains implementation along the idea of Deep Deterministic Policy Gradients:

- DDPG [[Paper](https://arxiv.org/abs/1509.02971)]

## Results

Plot shows scores per episode, being an average over all the agents, together with the threshold. Required average over the last 100 episodes for all the agents was set to be 30.

![scores](https://github.com/rrstal/drlnd/blob/master/continuous-control/media/result.png)