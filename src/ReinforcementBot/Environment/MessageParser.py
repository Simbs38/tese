import json
import re

'''
Message Color Meaning:
	<lightred>       - For messages you should most likely pay attention to
	<red>            - For messages about health or item destruction, or other very bad things
	<yellow>         - For messages you should probably pay attention to, includes when stuff makes noise
	<darkgrey>       - For messages you probably don't need to pay attention to
	<brown>          - For messages that describe damage to you or an ally
	<lightblue>      - For messages that describe you or an ally doing damage to an enemy
	<green>          - For messages that describe a beneficial effect that doesn't do damage
	<lightgreen>     - For messages that describe when something dies or something good occurs
	<lightmagenta>   - For messages that describe gaining good mutations
	<cyan>           - For messages that describe something to the player
	<blue>           - For messages that describe when either you or an enemy attacks/casts
	<magenta>        - For messages relating to Gods

'''

class MessageParser:
	def TryParseMessage(self, msg, dungeon):
		if 'msg' in msg:
			if msg['msg'] == 'msgs':
				self.ParseMessage(msg, dungeon)
			elif msg['msg'] == 'player':
				self.ParseUserInfo(msg, dungeon)
			
	def ParseMessage(self, msg, dungeon):
		if 'messages' in msg:
			for item in msg["messages"]:
				if "text" in item:
					if "ReinforcementStats" in item["text"]:
						self.ParseReinforcementStats(item["text"], dungeon)
						pass
					elif "Done exploring" in item["text"]:
						dungeon.ExploringDone = True
					elif "You climb" in item["text"]:
						dungeon.ExploringDone = False
					elif "You have reached level" in item["text"]:
						self.ParseLevel(item["text"], dungeon)
					'''
					elif "<lightblue>" in item["text"]:
						#print(item["text"]) # scrolls and potions that are on this space and can be picked up
						#check if is worth picking up the potions and scrolls
						pass
					elif "<green>" in item["text"]:
						#print(item["text"]) #items that are on this space and can be picked up
						#check if is worth picking up the item 
						pass
					'''


	def ParseUserInfo(self, msg, dungeon):
		print("here")
		if "species" in msg:
			dungeon.PlayerClass = msg["species"]
		if "hp" in msg:
			dungeon.Hp = msg["hp"]
		if "str" in msg:
			dungeon.Strength = msg["str"]
		if "int" in msg:
			dungeon.Intelligence = msg["int"]
		if "dex" in msg:
			dungeon.Dexterity = msg["dex"]
		if "xl" in msg:
			dungeon.Level = msg["xl"]
		if "progress" in msg:
			dungeon.LevelProgress = msg["progress"]
		if "turn" in msg:
			dungeon.Turns = msg["turn"]

	def ParseLevel(self, msg, dungeon):
		msgParts = re.split("[! ]", msg)
		try:
			level = int(msgParts[4])
			dungeon.Level = level
		except Exception as e:
			pass

	def ParseReinforcementStats(self, msg, dungeon):
		msgParts = re.split("[ <>]", msg)
		dungeon.PlayerRace = msgParts[3]
		dungeon.PlayerClass = msgParts[4]
		dungeon.Hp = msgParts[5]
		dungeon.Dexterity = msgParts[6]
		dungeon.Intelligence = msgParts[7]
		dungeon.Strength = msgParts[8]
		dungeon.HaveOrb = msgParts[ 9]
		dungeon.Hunger = msgParts[10]
		dungeon.Turns =  msgParts[11]
		dungeon.Where = msgParts[12]
		dungeon.LevelProgress = msgParts[13]