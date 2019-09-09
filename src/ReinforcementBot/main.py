from itertools import count
from Reinforcement import Utils, Agent, Dqn, Experience, QValues
from Environment import DungeonEnv, MapHandler, StateUpdateHandler
from GameConnection import GameConnection
from threading import Thread
import torch.nn.functional as F
import torch
import os

utils = Utils()

environment = DungeonEnv()
Thread(target=environment.MessageHandler.start).start()
Thread(target=GameConnection().start).start()

agent = Agent(utils, environment)


policyNet = Dqn(utils)

if os.path.isfile('./network.pth'):
	print("here")
	policyNet = torch.load('./network.pth')
	policyNet.eval()

targetNet = Dqn(utils)

utils.startOptimizer(policyNet)

targetNet.load_state_dict(policyNet.state_dict())
targetNet.eval()

episodesDuration = []

for episode in range(utils.NumEpisodes):
	environment.reset()
	state = environment.getState()

	for timestep in count():
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
			
		if environment.done:
			episodesDuration.append(timestep)

	if episode & utils.TargetUpdate == 0:
		targetNet.load_state_dict(policyNet.state_dict())

	print("episode end")
	torch.save(policyNet, './network.pth')


em.close()

print("done")
