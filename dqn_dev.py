import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import csv

#List of OpenAI Gym environments for testing
environments = ["CartPole-v1", "Acrobot-v1", "SpaceInvaders-ram-v0"]

discount_rate = 0.95

#List of learning rates to test
learning_rates = [0.001, 0.003, 0.005]

state_mem_max = 1000000
batch_size = 50

init_rand = 1.0
min_rand = 0.01
rand_decay = 0.999

episodes = 500000

#Implement class for DQN algorithm
class DQN_Alg:
	#Learning rate must be passed in as a parameter
	def __init__(self, obs_space, action_space, learning_rate):
		self.rand_rate = init_rand
		self.action_space = action_space
		self.learning_rate = learning_rate
		self.memory = deque(maxlen = state_mem_max)
		self.model = Sequential()
		self.model.add(Dense(48, input_shape=(obs_space,), activation="relu"))
		self.model.add(Dense(48, activation="relu"))
		self.model.add(Dense(self.action_space, activation="linear"))
		self.model.compile(loss="mse", optimizer=Adam(lr=learning_rate))

	#Append state to memory
	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	#Perform action given a state
	def act(self, state):
		#Perform random actions, depending random rate
		if np.random.rand() < self.rand_rate:
			return random.randrange(self.action_space)
		#Else act as to maximize predicted reward
		q_vals = self.model.predict(state)
		return np.argmax(q_vals[0])

	#Replay batch of history for training
	def replay(self):
		if len(self.memory) < batch_size:
			return
		batch = random.sample(self.memory, batch_size)
		for state, action, reward, next_state, done in batch:
			q = reward
			if not done:
				q = reward + discount_rate * np.amax(self.model.predict(next_state)[0])
			q_vals = self.model.predict(state)
			q_vals[0][action] = q
			self.model.fit(state, q_vals, verbose = 0)
		#Decrease random rate as learning occurs
		self.rand_rate *= rand_decay
		self.rand_rate = max(min_rand, self.rand_rate)

#Perform training on all environments with all learning rates
def play_envs():
	for environment in environments:
		for learning_rate in learning_rates:
			env = gym.make(environment)
			observation_space = env.observation_space.shape[0]
			action_space = env.action_space.n
			dqn = DQN_Alg(observation_space, action_space, learning_rate)
			results = []
			for run_num in range(episodes):
				state = env.reset()
				state = np.reshape(state, [1, observation_space])
				step_count = 0
				score = 0
				done = False
				while not done:
					env.render()
					#Have model perform an action
					action = dqn.act(state)
					#Update state based on the action
					next_state, reward, done, _ = env.step(action)
					score += reward
					#Negative reward if game is finished (game loss)
					if done:
						reward = -10
					#Set all positive rewards to 1 and all negative rewards to -1
					reward = np.sign(reward)
					next_state = np.reshape(next_state, [1, observation_space])
					dqn.remember(state, action, reward, next_state, done)
					state = next_state
					if done:
						results.append([run_num + 1, dqn.rand_rate, score])
						break
					dqn.replay()
			#Print results of training with a learning rate on a specific environment to a labeled file
			with open(environment+"_" + str(learning_rate) + ".csv", "w", newline = "") as file:
				writer = csv.writer(file)
				writer.writerows(results)

if __name__ == "__main__":
	play_envs()
