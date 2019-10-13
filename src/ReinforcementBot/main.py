from RunNetwork import RunNetwork as RN
from time import sleep
from Environment import DungeonEnv
from Reinforcement import Utils, Dqn1, Dqn2, Dqn3, Dqn4, Dqn5
from multiprocessing import Process
from GameConnection import GameConnection
from threading import Thread

utils = Utils()
environment = DungeonEnv(utils.WaitingTime)
environment.MessageConn = Thread(target=environment.StateUpdate.start)
environment.GameConn = Process(target=GameConnection().start)

environment.GameConn.start()
environment.MessageConn.start()

#################################################################### Teste1
'''
policyNet = Dqn2(utils)
targetNet = Dqn2(utils)
net = RN()

print("Network is about to start, please select the game tab")
sleep(5)
print("Network Starting")

net.run(utils, environment, policyNet, targetNet, "dqn2", "./DQN2")
'''
#################################################################### Teste2
'''
policyNet = Dqn3(utils)
targetNet = Dqn3(utils)
net = RN()

print("Network is about to start, please select the game tab")
sleep(5)
print("Network Starting")

net.run(utils, environment, policyNet, targetNet, "dqn3", "./DQN3MAP")


#################################################################### Teste3

policyNet = Dqn4(utils)
targetNet = Dqn4(utils)
net = RN()

print("Network is about to start, please select the game tab")
sleep(5)
print("Network Starting")

net.run(utils, environment, policyNet, targetNet, "dqn4", "./DQN4MAP")

#################################################################### Teste4

policyNet = Dqn5(utils)
targetNet = Dqn5(utils)

net = RN()

print("Network is about to start, please select the game tab")
sleep(5)
print("Network Starting")

net.run(utils, environment, policyNet, targetNet, "dqn5", "./DQN5MAP")


print("UPDATED LEARNING RATE")
'''
utils.LearningRate = 0.0001


policyNet = Dqn3(utils)
targetNet = Dqn3(utils)
net = RN()

print("Network is about to start, please select the game tab")
sleep(5)
print("Network Starting")

net.run(utils, environment, policyNet, targetNet, "dqn3", "./DQN3MAPLR1")


#################################################################### Teste3

policyNet = Dqn4(utils)
targetNet = Dqn4(utils)
net = RN()

print("Network is about to start, please select the game tab")
sleep(5)
print("Network Starting")

net.run(utils, environment, policyNet, targetNet, "dqn4", "./DQN4MAPLR1")

#################################################################### Teste4

policyNet = Dqn5(utils)
targetNet = Dqn5(utils)

net = RN()

print("Network is about to start, please select the game tab")
sleep(5)
print("Network Starting")

net.run(utils, environment, policyNet, targetNet, "dqn5", "./DQN5MAPLR1")



utils.LearningRate = 0.00001


policyNet = Dqn3(utils)
targetNet = Dqn3(utils)
net = RN()

print("Network is about to start, please select the game tab")
sleep(5)
print("Network Starting")

net.run(utils, environment, policyNet, targetNet, "dqn3", "./DQN3MAPLR2")


#################################################################### Teste3

policyNet = Dqn4(utils)
targetNet = Dqn4(utils)
net = RN()

print("Network is about to start, please select the game tab")
sleep(5)
print("Network Starting")

net.run(utils, environment, policyNet, targetNet, "dqn4", "./DQN4MAPLR2")

#################################################################### Teste4

policyNet = Dqn5(utils)
targetNet = Dqn5(utils)

net = RN()

print("Network is about to start, please select the game tab")
sleep(5)
print("Network Starting")

net.run(utils, environment, policyNet, targetNet, "dqn5", "./DQN5MAPLR2")
