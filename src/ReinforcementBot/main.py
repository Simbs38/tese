from RunNetwork import RunNetwork as RN
from time import sleep
from Environment import DungeonEnv
from Reinforcement import Utils, Dqn1, Dqn2
from multiprocessing import Process
from GameConnection import GameConnection
from threading import Thread

utils = Utils()
environment = DungeonEnv(utils.WaitingTime)
environment.MessageConn = Thread(target=environment.StateUpdate.start)
environment.GameConn = Process(target=GameConnection().start)

environment.GameConn.start()
environment.MessageConn.start()

policyNet = Dqn2(utils)
targetNet = Dqn2(utils)

net = RN()

print("Network is about to start, please select the game tab")
sleep(5)
print("Network Starting")

net.run(utils, environment, policyNet, targetNet, "dqn2", "./DQN2")

