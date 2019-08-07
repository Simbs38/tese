import random

class ReplayMemory:
	def __init__(self, utils):
		self.capacity = utils.MemorySize
		self.memory = []
		self.pushCount = 0

	def push(self, experience):
		if(len(self.memory) < self.capacity):
			self.memory.append(experience)
		else:
			self.memory[self.pushCount % self.capacity] = experience
		self.pushCount += 1

	def sample(self, batchSize):
		return random.sample(self.memory, batchSize)

	def canProvideSample(self, batchSize):
		return len(self.memory) >= batchSize