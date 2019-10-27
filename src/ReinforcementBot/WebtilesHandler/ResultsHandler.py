import os
import re
import matplotlib.pyplot as plt


def ParseRewards(folder, filename, reward):
	x = []
	y = []

	n = 1
	step = 20
	meanReward = 0
	while n < len(reward):
		tmp = int(reward[n]) - int(reward[n-1])
		if(tmp < 1000 and tmp > -600):
			meanReward += tmp
			
		if(n%step == 0):
			x.append(n-1 * step)
			y.append(meanReward/step)
			meanReward = 0
		n += 1	

	plt.plot(x,y)		

	plt.legend()
	plt.xlabel("Steps")
	plt.ylabel("Reward")
	plt.title("Rewards over time")

	plt.savefig(folder+"/"+ "REWARD" + filename[:-4]+".png")
	plt.close()


def ParseMoves(folder, filename, moves):
	x = []
	valid = []
	invalid = []
	n = 0
	
	if(len(moves)>0):
		base1 = int(moves[0][0]) 
		base2 = int(moves[0][1])

	while n < len(moves):
		valid.append(int(moves[n][0]) - base1)
		invalid.append(int(moves[n][1]) - base2)
		x.append(n * 100)
		n += 1

	plt.plot(x, valid, label = "Valid Actions")
	plt.plot(x, invalid, label = "Invalid Actions")

	plt.xlabel("Steps")
	plt.ylabel("Valid Actions")
	
	plt.title("Actions over time")
	plt.legend()

	plt.savefig(folder + "/" + "MOVES" + filename[:-4]+".png")
	plt.close()


def ParseLoss(folder, filename, loss):
	n = 0
	x = []
	y = []
	step = 50
	meanLoss = 0

	while n < len(loss):
		meanLoss += loss[n]
			
		if(n%step == 0):
			x.append(n-1 * step)
			y.append(meanLoss/step)
			meanLoss = 0
		n += 1	
	
	plt.plot(x,y)


	plt.legend()
	plt.xlabel("Steps")
	plt.ylabel("Loss")
	plt.title("Loss evolution over time")
	plt.savefig(folder+"/"+ "LOSS" + filename[:-4]+".png")
	plt.close()



def makeGraph(folder, filename):
	validMoves = []
	reward = []
	loss = []

	with open(folder + "/" + filename) as file:
		line = file.readline()
		while line:
			if(line[0] == 'R'):
				parts = re.split('[ ().]', line)
				reward.append(parts[2])
			if(line[0] == 'V'):
				parts = re.split('[ \n]', line)
				try:
					validMoves.append((parts[2], parts[3]))
				except Exception as e:
					pass
			if( "loss" in line):
				parts = re.split("[ \n]", line)
				loss.append(float(parts[2]))

			line = file.readline()

	ParseRewards(folder, filename, reward)
	ParseMoves(folder, filename, validMoves)
	ParseLoss(folder, filename, loss)

	print("Done: " + folder + "/" + filename)

def main():
	mainFolder = "results"
	for folder in os.listdir(mainFolder):
		for filename in os.listdir(mainFolder + "/" + folder):
			if(filename[-4:] == ".txt"):
				makeGraph(mainFolder + "/" + folder, filename)



if __name__ == "__main__":
	main()	