import json
import os

class ReaderFile:
	def __init__(self):
		self.json = None
		with open('messages.txt','r') as json_file:
			if os.path.getsize('messages.txt') > 0:
				self.json = json.loads(json_file.read())
				print("read")
				newList = []

				for msg in self.json:
					newList = self.AddListToList(newList, msg)

				for msg in newList:
					print(type(msg))
					if (msg['msg'] == 'map'):
						


	def AddListToList(self, newList, msg):
		if type(msg) is dict:
			newList.append(msg)
			return newList
		else:
			for item in msg:
				newList = self.AddListToList(newList, item)
			return newList