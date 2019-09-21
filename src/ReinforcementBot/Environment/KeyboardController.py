from pyautogui import press
from time import sleep


class KeyboardController:
	def __init__(self):
		self.actions = ['1','2','3','4','5','6','7','8','9','o','<','>','e']
		pass

	def ExecutAction(self, action, food):
		print("action " + self.actions[action.item()])
		press(self.actions[action.item()])
		if(self.actions[action] == 'e'):
			sleep(0.5)
			print("food at " + food)
			press(food)
			sleep(0.5)
			press('esc')

	def PressSpace(self):
		press("space")
		press("space")
		press("space")
		press("space")
		press("space")