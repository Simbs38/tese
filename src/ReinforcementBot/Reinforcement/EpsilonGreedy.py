import math

class EpsilonGreedyStrategy:
	def __init__(self, utils):
		self.start = utils.EpsStart
		self.end = utils.EpsEnd
		self.decay = utils.EpsDecay

	def getExplorationRate(self, currentStep):
		return self.end + (self.start - self.end) * math.exp(-1 * currentStep * self.decay)