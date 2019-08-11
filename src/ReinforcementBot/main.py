from itertools import count
from Reinforcement import Utils, Agent, Dqn, Experience
from Environment import DungeonEnv

utils = Utils()
environment = DungeonEnv()

agent = Agent(utils, environment)

policyNet = Dqn(utils)
targetNet = Dqn(utils)

utils.startOptimizer(policyNet)

targetNet.load_state_dict(policyNet.state_dict())
targetNet.eval()
environment = DungeonEnv()

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


			loss = F.mse_loss(currentQValues, targetQValues.unsqueeze(1))
			utils.optimizer.zero_grad()
			loss.backward()
			utils.optimizer.step()

		if environment.done:
			episodesDuration.append(timestep)

	if episode & utils.TargetUpdate == 0:
		targetNet.load_state_dict(policyNet.state_dict())

em.close()


'''

from MapHandler import MapHandler
import json
from StateUpdate import StateUpdateHandler
from pyautogui import press, typewrite, hotkey


if __name__ == "__main__":
    St = StateUpdateHandler()



#while(True):
#	press('a') script to write to keyboard and make game work




'''