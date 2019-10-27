# Playing DCSS with Deep Reinforcement Learning

## The Configuration File

This file must be placed on the webtiles server as your config file for the network to run.

This configuration file prints messages with relevant data to the message board, these messages are needed to update the state of our environment.

## Toml Config File

Besides that configuration file, you will also need to adjust the + [.toml file](https://github.com/Simbs38/tese/tree/master/src/ReinforcementBot/GameConnection/config.toml) that has some user specific configurations.

The filds that you need to change are:

 + username, that most be the acount that will whatch your games and serve as a communication bridge.
 + password, the password for the user that will whatch your games
 + server_url, the server where you will play
 + watch_player, your account name