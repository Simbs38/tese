
from Environment.MapHandler import MapHandler
from Environment.StateUpdate import StateUpdateHandler

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
		pass

	def reset(self):
		pass

	def render(self, close=False):
		pass

	def getState(self):
		pass

	def getActionCount(self):
		return 5