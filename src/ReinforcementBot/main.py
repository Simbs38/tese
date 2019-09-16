from RunNetwork import RunNetwork as RN
from time import sleep
from Environment import DungeonEnv
from Reinforcement import Utils, Dqn
from threading import Thread
from GameConnection import GameConnection

utils = Utils()
environment = DungeonEnv()
Thread(target=environment.MessageHandler.start).start()
Thread(target=GameConnection().start).start()

policyNet = Dqn(utils)
targetNet = Dqn(utils)

net = RN()

print("Network is about to start, please select the game tab")
sleep(5)
print("Network Starting")

net.run(utils, environment, policyNet, targetNet, "dqn1", "./networkDQN1.pth")

