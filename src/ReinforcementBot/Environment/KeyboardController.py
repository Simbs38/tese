from pyautogui import press
from time import sleep


class KeyboardController:
	def __init__(self):
		self.actions = ['1','2','3','4','5','6','7','8','9','o','<','>','e']
		pass

	def ExecutAction(self, action, food):
		press(self.actions[action.item()])
		if(self.actions[action] == 'e'):
			sleep(0.5)
			press(food)
			press('esc')

	def PressSpace(self):
		press("space")