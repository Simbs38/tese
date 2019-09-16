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
		self.ValidMoves = 0
		self.InvalidMoves = 0
		self.MessagesReceived = 0
		pass

	def step(self,action):
		tmpLevelProgress = self.LevelProgress
		tmpLevel = self.Level
		tmpTurns = self.Turns
		maxTryCount = 0

		while(tmpTurns == self.Turns):
			if(maxTryCount!=0):
				self.Keyboard.PressSpace()
			self.Keyboard.ExecutAction(action)
			self.actionCount = self.actionCount + 1

			while(self.MessagesReceived == 0):
				pass
			self.MessagesReceived = 0
			maxTryCount = maxTryCount + 1
			if(maxTryCount > 5):
				break


		ans = self.LevelProgress - tmpLevelProgress
		if(self.Level != tmpLevel):
			ans = (100+self.LevelProgress) - tmpLevelProgress

		if ans==0 and self.Turns == tmpTurns:
			ans = -1
			self.InvalidMoves = self.InvalidMoves + 1
		else:
			self.ValidMoves = self.ValidMoves + 1

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