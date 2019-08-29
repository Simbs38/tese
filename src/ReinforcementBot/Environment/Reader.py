import json
import os
from MessageParser import MessageParser
from DCSSEnv import DungeonEnv

class ReaderFile:
	def __init__(self):
		self.json = None
		self.dungeon = DungeonEnv()
		self.parser = MessageParser(self.dungeon)
		with open('messages.txt','r') as json_file:
			if os.path.getsize('messages.txt') > 0:
				self.json = json.loads(json_file.read())
				newList = []

				for msg in self.json:
					newList = self.AddListToList(newList, msg)

				for msg in newList:
					self.parser.TryParseMessage(msg)					

	def AddListToList(self, newList, msg):
		if type(msg) is dict:
			newList.append(msg)
			return newList
		else:
			for item in msg:
				newList = self.AddListToList(newList, item)
			return newList