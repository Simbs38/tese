from Environment.MapHandler import MapHandler
from Environment.StateUpdate import StateUpdateHandler
import numpy as np
import torch


class DungeonEnv:
	def __init__(self):
		self.done = False
		self.Hp = 0
		self.PlayerClass = ""
		self.PlayerRace = ""
		self.God = ""
		self.Dexterity = 0
		self.Intelligence = 0
		self.Strength = 0
		self.HaveOrb = False
		self.Hunger = 0
		self.Turns = 0
		self.Where = ""
		self.LevelProgress = 0
		self.Level = 0
		self.ExploringDone = False
		self.Map = MapHandler()
		self.MessageHandler = StateUpdateHandler(self)
		pass

	def step(self,action):
		return torch.tensor(0, dtype = torch.float32)

	def reset(self):
		self.done = False
		self.Hp = 0
		self.PlayerClass = ""
		self.PlayerRace = ""
		self.God = ""
		self.Dexterity = 0
		self.Intelligence = 0
		self.Strength = 0
		self.HaveOrb = False
		self.Hunger = 0
		self.Turns = 0
		self.Where = ""
		self.LevelProgress = 0
		self.Level = 0
		self.ExploringDone = False
		self.Map = MapHandler()
		pass

	def render(self, close=False):
		pass

	def getState(self):
		playerStats = [self.Hp, self.Dexterity, self.Intelligence, self.Strength, self.Hunger, self.HaveOrb, self.Turns, self.LevelProgress, self.Level, self.ExploringDone]
		mapState = self.Map.GetState(20,20)
		state = playerStats + mapState
		state = self.ClearState(state)
		state = np.array(state)
		ans = torch.from_numpy(state).type(torch.FloatTensor)
		return ans
		

	def getActionCount(self):
		return 5


	def ClearState(self, state):
		ans = []
		for item in state:
			if isinstance(item, int):
				ans.append(item)
			elif isinstance(item, list):
				for subitem in item:
					if isinstance(subitem, str):
						ans.append(ord(subitem[0]))
				pass
			else:
				print(type(item))

		return ans