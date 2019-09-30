# DQN Learning Rates
Project analyzing the effect of varying the learning rate of the neural network in the Deep Q Learning Algorithm. Tested using a number of OpenAI Gym environments.

## Overview

The purpose of this project is to analyze the effect of adjusting the learning rate of a neural net used in reinforcement learning. A Deep Q-Network is used here due to its proven ability to be effective in games and simulations as shown by DeepMind. The OpenAI Gym is used to supply a variety of standardized learning environments.

Three different learning environments were used: Cartpole-v1, Acrobot-v1, and SpaceInvaders-ram-v0. These environments were selected in order to provide different levels of complexity.

## Software Requirements

Python 3.6, Tensorflow 1.13.1, Keras 2.2.4, Gym, atari-py (https://github.com/Kojoley/atari-py)

## Summary of Results

For both cartpole and acrobot, the lowest learning rate performed best, and higher learning rates resulted in lower scores. Higher learning rates seem to to overtrain the model at points, not letting it slowly converge on a more effective strategy. The model struggled to learn on space invaders within the allotted time, as the environment is very complex.

## Model and Methods

Deep Q Networks (DQN) have been shown to be very effective at learning and improving within games. In this experiment, the goal was to test a DQN's performance while varying the learning rate of the neural net used to decide on the network's action at each point in time.

I based my implementation of DQN on a couple other implementations: https://github.com/gsurma/cartpole/blob/master/cartpole.py, https://github.com/keon/deep-q-learning. Some of the changes were made to improve the ability of the network to train on some of the more complex environments.

Cartpole-v1, Acrobot-v1, and SpaceInvaders-ram-v0 were selected as the testing environments as they provide varied levels of complexity in which to test the DQN. Cartpole is the simplest environment, with acrobot being slightly more difficult, and space invaders being the most complex.

Three learning rates (0.001, 0.003, and 0.005) were tested for each of the three environments, for a total of 9 runs. Each run consists of 500 episodes (playing the game until a loss occurs). The score for each episode is tracked.

## Results

See charts folder for data and graphs of results.

For the Cartpole runs, the lowest learning (0.001) clearly has the best scores, with a running average hovering around 150 from around 150 runs forward. The DQN learns quickly, and generally maintains a high average score. However, there is still a section from around runs 160-220 with very low scores. Reinforcement learning algorithms can occasionally reach areas where they have converged on an incorrect strategy, and take a while to adjust back. This is clearly seen here, as the model performs very badly for that period of time, but then recovers. As the learning rate is increased, the running average scores decrease. Additionally, as the learning rate is increased, the prevalence of sections with consistently low scores also increases.

Training on Acrobot shows similar patterns as Cartpole, with the lowest learning rate having by far the best results. The scores decrease as the learning rate increases, and there are again many areas with consistently lower scores with the higher learning rates. The model reached its highest scores more quickly (in terms of run number) with Acrobot than Cartpole, which is likely because the Acrobot environment tends to take slightly longer to run than Cartpole, especially at first. Thus, the model sees more states per run early on with Acrobot, resulting in more training and fewer random actions in earlier run numbers than with Cartpole.

Space Invaders did not seem to learn in any significant manner. There does not seem to be noticeable or consistent improvement above random actions for any of the learning rates throughout the entire training periods. As discussed earlier, Space Invaders is a far more complex environment than either Cartpole or Acrobot, so it does make sense that training struggles here. Given that the DQN model did improve on Cartpole and Acrobot, it is likely that it would also improve performance in Space Invaders given enough time, but 500 episodes does not seem to be enough.

## Conclusions

From the results, it seems that lower learning rates for neural nets tends to result in improved training for reinforcement learning. Due to the tendency of reinforcement learning models to struggle to converge, this makes sense, as a lower learning rate may then lead to a slow but steady path to convergence, whereas higher learning rates may lead to the neural net skipping over potential convergence areas. Along this line of thought, it would also be interesting to run similar experiments but with even lower learning rates.
