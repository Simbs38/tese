from time import sleep
from pyautogui import press, keyDown, keyUp, hotkey

class GameHandler:
	def __init__(self):
		pass

	def HandlerMessage(self, message):
		pass

	def ExitGame(self):
		sleep(4)
		press('space')
		press('esc')
		press('esc')
		sleep(1)
		press('esc')

	def RestartGame(self):
		self.ExitGame()
		self.StartNewRun()
		pass

	def StartNewRun(self):
		sleep(2)
		hotkey('shift','esc')
		press('f')
		press('e')
		hotkey('shift','esc')
		sleep(2)
		press('b') #for selecting minotaur
		sleep(2)
		press('h') #for selecting berzerker
		sleep(2)
		press('c') #for selecting weapon