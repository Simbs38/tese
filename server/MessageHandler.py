from json import loads
from time import sleep

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
		print("User " + str(type(msg)))

	def HandleMsgMap(self, msg):
		print("Map " + str(type(msg)))
		print(msg)

	def HandleMsgMsgs(self, msg):
		print("Msgs " + str(type(msg)))