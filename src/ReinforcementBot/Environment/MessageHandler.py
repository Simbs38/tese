from Environment.MessageParser import MessageParser
from json import load, loads, dumps
from time import sleep, time
import os    

class MessageHandler():
	messagesReceived = None

	def __init__(self, dungeon):
		self.messagesReceived = []
		self.dungeon = dungeon
		self.parser = MessageParser(dungeon)

	def ReceiveMsg(self, msg):
		data = msg.decode('utf-8')
		messageParsed = False
		startTime = time()
		while not messageParsed:
			if time() - startTime > 4 :
				break
			try:
				dataDict = loads(data)
				self.HandleMsg(dataDict)
				messageParsed = True
			except Exception as e:
				pass
				
	def HandleMsg(self, dataDict):
		for key, value in dataDict.items():
			if((key == "username") and (value == "simbs38") or (key == "name") and (value == "simbs38")):
				self.parser.ParseUserInfo(dataDict)
				break
			elif (value == "map"):
				self.dungeon.Map.UpdateMap(dataDict)
				self.dungeon.MessagesReceived = self.dungeon.MessagesReceived + 1
				break
			elif ( value == "msgs"):
				self.parser.ParseMessage(dataDict)
				break