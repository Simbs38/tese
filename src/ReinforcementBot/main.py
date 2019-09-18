from RunNetwork import RunNetwork as RN
from time import sleep
from Environment import DungeonEnv
from Reinforcement import Utils, Dqn1, Dqn2
from threading import Thread
from multiprocessing import Process
from GameConnection import GameConnection

utils = Utils()
environment = DungeonEnv(utils.WaitingTime)
Thread(target=environment.MessageHandler.start).start()
environment.GameConn = Process(target=GameConnection().start)
environment.GameConn.start()

policyNet = Dqn2(utils)
targetNet = Dqn2(utils)

net = RN()

print("Network is about to start, please select the game tab")
sleep(5)
print("Network Starting")

net.run(utils, environment, policyNet, targetNet, "dqn2", "./networkDQN2.pth")

