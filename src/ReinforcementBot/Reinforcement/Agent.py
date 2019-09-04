import random
import torch

class Agent:
	def __init__(self, utils, env):
		self.currentStep = 0
		self.strategy = utils.Strategy
		self.numActions = env.getActionCount()
		self.device = utils.device

	def selectAction(self, state, policyNet):
		rate = self.strategy.getExplorationRate(self.currentStep)
		self.currentStep +=1

		if rate > random.random():
			action = random.randrange(self.numActions)
			return torch.tensor(action).to(self.device)
		else:
			with torch.no_grad():
				return policyNet(state).argmax().to(self.device)


