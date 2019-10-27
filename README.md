# Playing DCSS with Deep Reinforcement Learning

The project consists in the bot that is able to play DCSS through Deep Reinforcement Learning.

The project was made to run under localy using the webtiles version of the game, the server used and the other settings can be defined in the [config.toml](https://github.com/Simbs38/tese/blob/master/src/GameConnection/config.toml) file.

## Running the project

To run the project the user must:
+ Edit the config.toml file
+ Copy the content of the [.rc file](https://github.com/Simbs38/tese/blob/master/src/configs.rc) to the .rc file in use
+ Install all dependencys
+ Run the make command from the main folder

## Dependencys




## The MakeFile

To run the network, run the command make, to analyse the results, run the comand make res, after inserting all of the results in the folder results

## Project Arquitecture

The project consists of several packages that work together. 

### ReinforcementBot

There are 4 main packages in the project:

+ [Reinforcement](https://github.com/Simbs38/tese/tree/master/src/ReinforcementBot/Reinforcement)
+ [Environment](https://github.com/Simbs38/tese/tree/master/src/ReinforcementBot/Environment).
+ [GameConnection](https://github.com/Simbs38/tese/tree/master/src/ReinforcementBot/GameConnection)
+ [WebtilesHanlder](https://github.com/Simbs38/tese/tree/master/src/ReinforcementBot/WebtilesHandler)

The Reinforcement handles all the classes that are needed to run deep reinforcement learning, and the Environment deals with the game state in every step of the game.

The main file uses the classes in the Reinforcement and Environment packages, in order to run the network under our build environment. 
To update the environment, a thread is created to listen to messages sent from the game, this messages are parsed and update the state of the environment, and from this we can improve our network and learn how to play the game.

## Project Background

The GameConnection package is simplification of the [Beem project](https://github.com/gammafunk/beem), where we just use the code that gets us the game information as 


This project was made in colaboration with [Universidade do Minho](https://www.uminho.pt/PT) in the [Informatics Department](https://www.di.uminho.pt/) as a Masters Degrees Thesis in the area of Artificial Inteligence.
