# Playing DCSS with Deep Reinforcement Learning

## Environment Package

The environment package is composed of the environment of the game itself, this class will keep the information about the current state of the game.
Besides that, this class also keeps the messages maps handler and the state updater.

This state update runs in a thread to receive messages from the game connection package and update the environment itself.

Finally, this package also includes a class that sends the our commands to the game.