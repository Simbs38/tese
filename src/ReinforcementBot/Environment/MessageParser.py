import json

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
		else:
			self.ParseUserInfo(msg, dungeon)


	def ParseMessage(self, msg, dungeon):
		if 'messages' in msg:
			for item in msg["messages"]:
				if "text" in item:
					if "<lightred>" in item["text"]:
						pass
					elif "<red>" in item["text"]:
						pass
					elif "<yellow>" in item["text"]:
						pass
					elif "<brown>" in item["text"]:
						print(item["text"]) # no messages found with this color
						pass
					elif "<lightblue>" in item["text"]:
						#print(item["text"]) # scrolls and potions that are on this space and can be picked up
						#check if is worth picking up the potions and scrolls
						pass
					elif "<green>" in item["text"]:
						#print(item["text"]) #items that are on this space and can be picked up
						#check if is worth picking up the item 
						pass
					elif "<lightgreen>" in item["text"]:
						#print(item["text"])
						pass
					elif "<lightmagenta>" in item["text"]:
						#print(item["text"])
						pass
					elif "<cyan>" in item["text"]:
						#print(item["text"])
						pass
					elif "<blue>" in item["text"]:
						#print(item["text"])
						pass
					elif "<magenta>" in item["text"]:
						#print(item["text"])
						pass
					elif "<darkgrey>" in item["text"]:
						pass #messages that can be ignored about combat info
					elif "<lightgrey>" in item["text"]:
						#print(item["text"])
						pass
					else:
						print(item["text"])

	def ParseUserInfo(self, msg, dungeon):
		pass