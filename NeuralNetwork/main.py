import sys
sys.path.insert(1,'Nets')
from Dqn1 import Dqn
from Utils import Utils, Experience
from Agent import Agent
from itertools import count

utils = Utils()
agent = Agent(utils)

policyNet = Dqn(utils)
targetNet = Dqn(utils)

utils.startOptimizer(policyNet)

targetNet.load_state_dict(policyNet.state_dict())
targetNet.eval()

episodesDuration = []

for episode in range(utils.NumEpisodes):
	utils.Env.reset()
	state = utils.Env.getState()

	for timestep in count():
		action = agent.selectAction(state, policyNet)
		reward = utils.Env.step(action)
		nextState = utils.Env.getState()

		utils.Memory.push(Experience(state, action, nextState, reward))
		state = nextState

		if(utils.Memory.canProvideSample(utils.BatchSize)):
			experiences = utils.Memory.sample(utils.BatchSize)
			states, actions, rewards, nextStates = utils.extractTensors(experiences)

			currentQValues = QValues.getCurrent(policyNet, states, actions)
			nextQValues = QValues.getNext(targetNet,nextStates, utils)
			targetQValues = (nextQValues * utils.Gamma) + rewards


			loss = F.mse_loss(currentQValues, targetQValues.unsqueeze(1))
			utils.optimizer.zero_grad()
			loss.backward()
			utils.optimizer.step()

		if utils.Env.done:
			episodesDuration.append(timestep)

	if episode & utils.TargetUpdate == 0:
		targetNet.load_state_dict(policyNet.state_dict())

em.close()