from Environment.MapHandler import MapHandler
from Environment.StateUpdate import StateUpdateHandler
from Environment.KeyboardController import KeyboardController
import numpy as np
import torch
import time
from multiprocessing import Process
from threading import Thread
from GameConnection import GameConnection

class DungeonEnv:
	def __init__(self, waitingTime):
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
		self.Keyboard = KeyboardController()
		self.actionCount = 0
		self.ValidMoves = 0
		self.InvalidMoves = 0
		self.MessagesReceived = 0
		self.WaitingTime = waitingTime
		self.GameConn = None
		self.MessageConn = None
		self.FoodKey = ''
		self.StateUpdate = StateUpdateHandler(self)
		self.ResetCount = 0
		pass

	def step(self,action):
		tmpLevelProgress = self.LevelProgress
		tmpLevel = self.Level
		tmpTurns = self.Turns
		tmpMap = self.Map.GetMapExploration()
		tmpMapDepth = self.Map.currentLevel
		tmpHp = self.Hp
		
		self.Keyboard.PressSpace()

		self.Keyboard.ExecutAction(action, self.FoodKey, self.Turns)
		self.actionCount = self.actionCount + 1
		
		time.sleep(1)
		
		startTime = time.time()
		while(self.MessagesReceived == 0):
			if(time.time() - startTime > self.WaitingTime):
				self.ResetCount = self.ResetCount + 1
				self.GameConn.terminate()
				self.GameConn.join()
				self.StateUpdate.stop()
				time.sleep(5)
				self.GameConn = Process(target=GameConnection().start)
				self.StateUpdate = StateUpdateHandler(self)
				self.MessageConn = Thread(target= self.StateUpdate.start)
				self.GameConn.start()
				self.MessageConn.start()
				break
		
		if(time.time() - startTime < self.WaitingTime):
			self.ResetCount = 0

		self.MessagesReceived = 0

		ans = self.GetReward(tmpLevelProgress, tmpLevel, tmpTurns, tmpMap, tmpMapDepth ,tmpHp)

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
		playerStats = [self.Hp, self.Dexterity, self.Intelligence, self.Strength, self.Hunger, self.HaveOrb, int(self.Turns), self.LevelProgress, self.Level, self.ExploringDone]
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

	def GetReward(self,tmpLevelProgress, tmpLevel, tmpTurns, tmpMap, tmpMapDepth, tmpHp):

		#Consider player xp level
		ans = self.LevelProgress - tmpLevelProgress
		if(self.Level != tmpLevel):
			ans = (100+self.LevelProgress) - tmpLevelProgress

		#Consider player hp
		ans = ans - (tmpHp - self.Hp)

		#Consider player hunger
		if (self.Hunger < 4):
			ans = ans - (4 - self.Hunger)

		#Consider if the player made or not a valid mode in the game
		if ans==0 and self.Turns == tmpTurns:
			ans = -1
			self.InvalidMoves = self.InvalidMoves + 1
		else:
			self.ValidMoves = self.ValidMoves + 1

		#Consider if the player explored the map or not, and if it was went deeper into the dungeon
		if tmpMapDepth != self.Map.currentLevel:
			ans = ans + 100
		else:
			ans = ans + (tmpMap - self.Map.GetMapExploration())

		return ans