from pyautogui import press

class KeyboardController:
	def __init__(self):
		self.actions = ['1','2','3','4','5','6','7','8','9','o','<','>']
		pass

	def ExecutAction(self, action):
		print(self.actions[action.item()], action)
		press(self.actions[action.item()])
