from collections import namedtuple
from EpsilonGreedy import EpsilonGreedyStrategy
from DCSSEnv import DungeonEnv as dcss
from ReplayMemory import ReplayMemory
import torch
import torch.optim as optim



Experience = namedtuple(
	'Experience',
	('state', 'action', 'next_state', 'reward')
)

class Utils:
	def __init__(self):
		self.BatchSize = 50
		self.Gamma = 0.999
		self.EpsStart = 1
		self.EpsEnd = 0.01
		self.EpsDecay = 0.001
		self.TargetUpdate = 10
		self.MemorySize = 100000
		self.LearningRate = 0.001
		self.NumEpisodes = 1000
		self.InputSize = 256 #check this after
		self.OutputSize = 10

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.Strategy = EpsilonGreedyStrategy(self)
		self.Env = dcss()
		self.Memory = ReplayMemory(self)

	def startOptimizer(self, policyNet):
		self.optimizer = optim.Adam(params=policyNet.parameters(), lr=self.LearningRate)

	def extractTensors(self, experiences):
		batch = Experience(*zip(*experiences))

		t1 = torch.cat(batch.state)
		t2 = torch.cat(batch.action)
		t3 = torch.cat(batch.reward)
		t4 = torch.cat(batch.next_state)

		return (t1,t2,t3,t4)