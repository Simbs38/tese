from itertools import count
from Reinforcement import Utils, Agent, Dqn, Experience, QValues
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
		pass

	def run(self, utils, environment, policyNet, targetNet, networkName, networkPath):
		environment.reset()
		ga = GameHandler()
		agent = Agent(utils, environment)


		if os.path.isfile(networkPath):
			policyNet = torch.load(networkPath)
			policyNet.eval()

		utils.startOptimizer(policyNet)

		targetNet.load_state_dict(policyNet.state_dict())
		targetNet.eval()

		networkLoss = []
		state = environment.getState()
		episode = 0
		
		while(episode < utils.NumEpisodes and not environment.done):

			action = agent.selectAction(state, policyNet)
			reward = environment.step(action)
			nextState = environment.getState()
			utils.Memory.push(Experience(state, action, nextState, reward))
			state = nextState

			if(utils.Memory.canProvideSample(utils.BatchSize)):
				experiences = utils.Memory.sample(utils.BatchSize)
				states, actions, rewards, nextStates = utils.extractTensors(experiences)

				currentQValues = QValues.getCurrent(policyNet, states, actions)
				nextQValues = QValues.getNext(targetNet,nextStates, utils)
				targetQValues = (nextQValues * utils.Gamma) + rewards

				targetQValues = targetQValues.unsqueeze(1)
				currentQValues = currentQValues
				
				loss = F.mse_loss(currentQValues, targetQValues)
				utils.optimizer.zero_grad()
				loss.backward()
				utils.optimizer.step()

				print("| " + str(episode) + " loss: " + str(loss.item()), end =' ')

			if episode%utils.TargetUpdate == 0:
				targetNet.load_state_dict(policyNet.state_dict())

			episode = episode + 1
			print("|" + str(episode), end = ' ')

			if episode%100 == 0:
				named_tuple = time.localtime()
				time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
				networkLoss.append([time_string,loss])

			if episode % 1000 == 0:
				torch.save(policyNet, networkPath)

			if environment.done:
				print("DIIIIEEE IN MAIN " + "Valid Moves: " + str(environment.ValidMoves) + " " + str(environment.InvalidMoves))
				environment.reset()
				time.sleep(4)
				print("restarting")
				ga.RestartGame()
				
		print(str(episode) + " "  + str(utils.NumEpisodes) + " " +str(environment.done) )
		named_tuple = time.localtime()
		time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
		networkLoss.append([time_string,loss])

		print("Done " + networkName)
