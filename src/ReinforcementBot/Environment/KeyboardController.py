from pyautogui import press
from time import sleep


class KeyboardController:
	def __init__(self):
		self.actions = ['1','2','3','4','5','6','7','8','9','o','e']
		pass

	def ExecutAction(self, action, food, turns):
		print("action " + self.actions[action.item()] + " at Turn:" + str(turns))
		press(self.actions[action.item()])
		if(self.actions[action] == 'e'):
			print("food at " + food)
			sleep(0.5)
			press(food)
			sleep(0.5)
			press('esc')

		return self.actions[action]

	def PressSpace(self):
		press("space")
		
	def UpgradeStats(self):
		press('S')

	def GoDownStairs(self):
		press('G')
		press('>')