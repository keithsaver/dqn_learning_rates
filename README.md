# DQN Learning Rates
Project analyzing the effect of varying the learning rate of the neural network in the Deep Q Learning Algorithm. Tested using a number of OpenAI Gym environments.

## Overview

The purpose of this project is to analyze the effect of adjusting the learning rate of a neural net used in reinforcement learning. A Deep Q-Network is used here due to its proven ability to be effective in games and simulations as shown by DeepMind. The OpenAI Gym is used to supply a variety of standardized learning environments.

Three different learning environments were used: Cartpole-v1, Acrobot-v1, and SpaceInvaders-ram-v0. These environments were selected in order to provide different levels of complexity.

## Software Requirements

Python 3.6, Tensorflow 1.13.1, Keras 2.2.4, Gym, atari-py (https://github.com/Kojoley/atari-py)

## Summary of Results

For both cartpole and acrobot, the lowest learning rate performed best, and higher learning rates resulted in lower scores. Higher learning rates seem to to overtrain the model at points, not letting it slowly converge on a more effective strategy. The model struggled to learn on space invaders, as the environment is very complex (even with 15+ hours of training, with following hardware: Windows 10 with i7 - 6700k CPU @ 4.00GHz, 32GB RAM).

## Model and Methods

Deep Q Networks (DQN) have been shown to be very effective at learning and improving within games. In this experiment, the goal was to test a DQN's performance while varying the learning rate of the neural net used to decide on the network's action at each point in time.

I based my implementation of DQN on a couple other implementations: https://github.com/gsurma/cartpole/blob/master/cartpole.py, https://github.com/keon/deep-q-learning. Some of the changes were made to improve the ability of the network to train on some of the more complex environments.

Cartpole-v1, Acrobot-v1, and SpaceInvaders-ram-v0 were selected as the testing environments as they provide varied levels of complexity in which to test the DQN. Cartpole is the simplest environment, with acrobot being slightly more difficult, and space invaders being the most complex.

Three learning rates (0.001, 0.003, and 0.005) were tested for each of the three environments, for a total of 9 runs. Each run consists of 500 episodes (playing the game until a loss occurs). The score for each episode is tracked.

## Results
