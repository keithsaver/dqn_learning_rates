# DQN Learning Rates
Project analyzing the effect of varying the learning rate of the neural network in the Deep Q Learning Algorithm. Tested using a number of OpenAI Gym environments.

## Overview

The purpose of this project is to analyze the effect of adjusting the learning rate of a neural net used in reinforcement learning. A Deep Q-Network is used here due to its proven ability to be effective in games and simulations as shown by DeepMind. The OpenAI Gym is used to supply a variety of standardized learning environments.

Three different learning environments were used: Cartpole-v1, Acrobot-v1, and SpaceInvaders-ram-v0. These environments were selected in order to provide different levels of complexity.

## Software Requirements

Python 3.6, Tensorflow 1.13.1, Keras 2.2.4, Gym, atari-py (https://github.com/Kojoley/atari-py)

## Summary of Results

For both cartpole and acrobot, the lowest learning rate performed best, and higher learning rates resulted in lower scores. Higher learning rates seem to to overtrain the model at points, not letting it slowly converge on a more effective strategy. The model struggled to learn on space invaders, as the environment is very complex (even with 15+ hours of training, with following hardware: Windows 10 with i7 - 6700k CPU @ 4.00GHz, 32GB RAM).

