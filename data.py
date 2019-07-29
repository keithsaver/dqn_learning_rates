import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


filenames = ["CartPole-v1_0.001.csv", "CartPole-v1_0.003.csv", "CartPole-v1_0.005.csv", \
			 "Acrobot-v1_0.001.csv", "Acrobot-v1_0.003.csv", "Acrobot-v1_0.005.csv", \
			 "SpaceInvaders-ram-v0_0.001.csv", "SpaceInvaders-ram-v0_0.003.csv", "SpaceInvaders-ram-v0_0.005.csv"]

envs = ["Cartpole", "Acrobot", "Space Invaders"]

learning_rates = ["0.001", "0.003", "0.005"]

#Read data from CSV to list of lists
def csv_to_list(file):
	data = []
	with open(file, newline = "") as f:
		reader = csv.reader(f)
		for row in reader:
			data.append(row)
	return data

#Calculate running averages, append to each row
def running_avgs(data):
	sum_scores = 0
	for row_num in range(len(data)):
		sum_scores += float(data[row_num][2])
		data[row_num].append(sum_scores/(row_num + 1))
	return data

#Print scores and running average per run for each environment learning rate pair
def print_scores(data, num):
	run_num = [float(row[0]) for row in data]
	scores = [float(row[2]) for row in data]
	avgs = [row[3] for row in data]
	
	env = envs[int(num/3)]
	lrate = learning_rates[num%3]

	f, ax = plt.subplots(1, 2, figsize = (10, 5))
	ax[0].scatter(run_num, scores, marker = ".")
	ax[0].set_title(env +', Learning Rate ' + lrate)
	ax[0].xaxis.set_major_locator(ticker.MultipleLocator(50))
	ax[0].yaxis.set_major_locator(ticker.MultipleLocator(50))
	ax[0].set_xlabel('Run Number')
	ax[0].set_ylabel('Score')
	ax[1].plot(run_num, avgs)
	ax[1].set_title(env +' Running Average, Learning Rate ' + lrate)
	ax[1].xaxis.set_major_locator(ticker.MultipleLocator(50))
	ax[1].yaxis.set_major_locator(ticker.MultipleLocator(20))
	ax[1].set_xlabel('Run Number')
	ax[1].set_ylabel('Score Running Average')
	#plt.show()
	plt.savefig(env + lrate + "_graphs.png")

#Run for all CSV's
for i in range(len(filenames)):
	data = csv_to_list(filenames[i])
	data = running_avgs(data)
	print_scores(data, i)