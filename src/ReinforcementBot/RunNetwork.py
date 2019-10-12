from itertools import count
from Reinforcement import Utils, Agent, Experience, QValues
from Environment import DungeonEnv, MapHandler, StateUpdateHandler
from GameConnection import GameConnection
from WebtilesHandler import GameHandler
from threading import Thread
import torch.nn.functional as F
import torch
import os
import time


class RunNetwork():
	def __init__(self):
		self.totalReward = 0
		pass

	def run(self, utils, environment, policyNet, targetNet, networkName, networkDir):
		if( not os.path.exists(networkDir)):
			os.makedirs(networkDir)

		networkPath = networkDir + "/network.pth"
		fileTime = time.localtime()
		stringTime = time.strftime("%m_%d_%Y_%H_%M", fileTime)
		lossPath = networkDir + "/loss-" + stringTime + ".txt"
		lossFile = open(lossPath, "w+")

		ga = GameHandler()
		agent = Agent(utils, environment)

		if os.path.isfile(networkPath):
			policyNet = torch.load(networkPath)
			policyNet.eval()

		utils.startOptimizer(policyNet)

		targetNet.load_state_dict(policyNet.state_dict())
		targetNet.eval()

		state = environment.getState()
		episode = 0
		
		while(episode < utils.NumEpisodes and not environment.done):

			action = agent.selectAction(state, policyNet)
			reward = environment.step(action)
			self.totalReward = self.totalReward + reward
			nextState = environment.getState()
			
			if(environment.ResetCount > 5):
				break

			utils.Memory.push(Experience(state, action, nextState, reward))
			state = nextState

			if(utils.Memory.canProvideSample(utils.BatchSize)):
				experiences = utils.Memory.sample(utils.BatchSize)
				states, actions, rewards, nextStates = utils.extractTensors(experiences)

				currentQValues = QValues.getCurrent(policyNet, states, actions)
				nextQValues = QValues.getNext(targetNet,nextStates, utils)
				targetQValues = (nextQValues * utils.Gamma) + rewards

				targetQValues = targetQValues.unsqueeze(1)
				
				loss = F.mse_loss(currentQValues, targetQValues)
				utils.optimizer.zero_grad()
				loss.backward()
				utils.optimizer.step()

				lossFile.write(str(episode) + " loss: " + str(loss.item()) + "\n")
				lossFile.write("Reward: " + str(self.totalReward) + "\n")
				print(str(episode) + " loss: " + str(loss.item()))


			if episode%utils.TargetUpdate == 0:
				targetNet.load_state_dict(policyNet.state_dict())

			episode = episode + 1

			print(episode)
			
			if episode%100 == 0:
				named_tuple = time.localtime()
				time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
				lossFile.write("Valid Moves: " + str(environment.ValidMoves) + " " + str(environment.InvalidMoves) + " " + time_string+ "\n")

			if episode % 1000 == 0:
				torch.save(policyNet, networkPath)

			if environment.done:
				named_tuple = time.localtime()
				time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
				lossFile.write("Valid Moves: " + str(environment.ValidMoves) + " " + str(environment.InvalidMoves) + " " + time_string + "\n")
				lossFile.write("##############################################" + "\n")
				lossFile.write("##############################################" + "\n")
				lossFile.write("##############################################" + "\n")
				print("Valid Moves: " + str(environment.ValidMoves) + " " + str(environment.InvalidMoves) + "\n")
				environment.reset()
				time.sleep(4)
				print("restarting")
				ga.RestartGame()
				
		print(str(episode) + " "  + str(utils.NumEpisodes) + " " +str(environment.done) )
		named_tuple = time.localtime()
		time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)

		lossFile.close()
		print("Done " + networkName)
