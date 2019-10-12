import json
import re

class MessageParser:
	def __init__(self, dungeon):
		self.dungeon = dungeon

	def TryParseMessage(self, msg):
		if 'msg' in msg:
			if msg['msg'] == 'msgs':
				self.ParseMessage(msg)
			elif msg['msg'] == 'player':
				self.ParseUserInfo(msg)
			
	def ParseMessage(self, msg):
		if 'messages' in msg:
			for item in msg["messages"]:
				if "text" in item:
					if "ReinforcementStats" in item["text"]:
						self.ParseReinforcementStats(item["text"])
						self.dungeon.MessagesReceived = self.dungeon.MessagesReceived + 1
					if "InventoryStats" in item["text"]:
						self.ParseInventory(item["text"])
					elif "Done exploring" in item["text"]:
						self.dungeon.ExploringDone = True
					elif "You climb" in item["text"]:
						self.dungeon.ExploringDone = False
						if "downwards" in item["text"]:
							self.dungeon.Map.LowerOneLevel()
						else:
							self.dungeon.Map.UpOneLevel()
					elif "You have reached level" in item["text"]:
						self.ParseLevel(item["text"])
					elif "Increase (S)trength, (I)ntelligence, or (D)exterity?" in item["text"]:
						self.IncreaseLevel() 
					if "You die" in item["text"]:
						self.dungeon.done = True

	def ParseUserInfo(self, msg):
		if "species" in msg:
			self.dungeon.PlayerClass = msg["species"]
		if "hp" in msg:
			self.dungeon.Hp = msg["hp"]
		if "str" in msg:
			self.dungeon.Strength = int(msg["str"])
		if "int" in msg:
			self.dungeon.Intelligence = int(msg["int"])
		if "dex" in msg:
			self.dungeon.Dexterity = int(msg["dex"])
		if "xl" in msg:
			self.dungeon.Level = int(msg["xl"])
		if "progress" in msg:
			self.dungeon.LevelProgress = int(msg["progress"])
		if "turn" in msg:
			self.dungeon.Turns = float(msg["turn"])

	def ParseLevel(self, msg):
		msgParts = re.split("[! ]", msg)
		try:
			level = int(msgParts[4])
			self.dungeon.Level = level
		except Exception as e:
			pass

	def ParseReinforcementStats(self, msg):
		msgParts = re.split("[ <>]", msg)
		self.dungeon.PlayerRace = msgParts[3]
		self.dungeon.PlayerClass = msgParts[4]
		self.dungeon.Hp = int(msgParts[5])
		self.dungeon.Dexterity = int(msgParts[6])
		self.dungeon.Intelligence = int(msgParts[7])
		self.dungeon.Strength = int(msgParts[8])
		if(msgParts[9][0] == 'f' or msgParts[9][0] == 'F'):
			self.dungeon.HaveOrb = False
		else:
			self.dungeon.HaveOrb = True
		self.dungeon.Hunger = int(msgParts[10])
		self.dungeon.Turns =  int(msgParts[11])
		self.dungeon.Where = msgParts[12]
		self.dungeon.LevelProgress = int(msgParts[13])

	def ParseInventory(self, msg):
		tmpFood = ''
		msgParts = re.split("[<>,]", msg)
		for item in msgParts:
			if("InventoryStats" in item):
				tmpParts = re.split(" ",item)
				if("food" in tmpParts[2]):
					tmpFood = tmpParts[1][0]
					break
		self.dungeon.FoodKey = tmpFood

	def IncreaseLevel(self):
		dungeon.ChooseStatToUpgrade()