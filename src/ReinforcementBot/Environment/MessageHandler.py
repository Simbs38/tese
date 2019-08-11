from json import load, loads, dumps
from time import sleep
import os    

class MessageHandler():
	messagesReceived = None

	def __init__(self):
		self.messagesReceived = []

	def ReceiveMsg(self, msg):
		data = msg.decode('utf-8')
		sleep(.05) #sleep a bit in order to give json.load time to read the previus message
		dataDict = loads(data)
		self.HandleMsg(dataDict)

	def HandleMsg(self, dataDict):
		for key, value in dataDict.items():
			if((key == "username") and (value == "simbs38") or (key == "name")     and (value == "simbs38")):
				print("user")
				self.HandleMsgUser(dataDict)
				break
			elif (value == "map"):
				print("map")
				self.HandleMsgMap(dataDict)
				break
			elif ( value == "msgs"):
				print("msgs")
				self.HandleMsgMsgs(dataDict)
				break

	def HandleMsgUser(self, msg):
		self.GeneralHandler(msg)
		print("User " + str(type(msg)))

	def HandleMsgMap(self, msg):
		self.GeneralHandler(msg)
		print("Map " + str(type(msg)))

	def HandleMsgMsgs(self, msg):
		self.GeneralHandler(msg)
		print("Messages " + str(type(msg)))


	def GeneralHandler(self, msg):
		oldData = None
		with open('messages.txt','r') as json_file:
			if os.path.getsize('messages.txt') > 0:
				oldData = loads(json_file.read())
		with open('messages.txt','w+') as json_file:
			if oldData == None:
				oldData = []
			else:
				oldData = [oldData]
			oldData.append(msg)
			jsoned_data = dumps(oldData, indent = True)
			json_file.write(jsoned_data)

