from Environment.MapHandler import MapHandler
from Environment.StateUpdate import StateUpdateHandler
from Environment.KeyboardController import KeyboardController
import numpy as np
import torch

from time import sleep


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
		self.Keyboard = KeyboardController()
		self.actionCount = 0
		pass

	def step(self,action):
		tmpLevelProgress = self.LevelProgress
		tmpLevel = self.Level

		self.Keyboard.ExecutAction(action)
		self.actionCount = self.actionCount + 1
		sleep(1)

		ans = self.LevelProgress - tmpLevelProgress
		if(self.Level != tmpLevel):
			ans = (100+self.LevelProgress) - tmpLevelProgress

		return torch.tensor(ans, dtype = torch.float32)

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

	def getState(self):
		playerStats = [self.Hp, self.Dexterity, self.Intelligence, self.Strength, self.Hunger, self.HaveOrb, self.Turns, self.LevelProgress, self.Level, self.ExploringDone]
		mapState = self.Map.GetState(20,20)
		state = playerStats + mapState
		state = self.ClearState(state)
		state = np.array(state)
		ans = torch.from_numpy(state).type(torch.FloatTensor)

		return ans
		

	def getActionCount(self):
		return self.actionCount

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

		return ans