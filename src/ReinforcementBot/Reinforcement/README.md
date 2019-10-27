# Playing DCSS with Deep Reinforcement Learning

## Game Connection Package

The Reinforcement package is the package that holds all of the networks that were tested during this project. 

This package also includes other classes that are important to train the network, such as the agent, that will run the network at each step of the game to get the action to execute, the epsilon greedy strategy, that decides if the network should make a decision about what action to play, or if it should play an random action in order to explore the environment and therefore, gain more knowledge of it. It also includes our replay memory where we are going to save our experiences during the training of the network.