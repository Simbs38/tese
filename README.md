# Playing DCSS with Deep Reinforcement Learning

The project consists in the bot that is able to play DCSS through Deep Reinforcement Learning.

The project was made to run under localy using the webtiles version of the game, the server used and the other settings can be defined in the [config.toml](https://github.com/Simbs38/tese/blob/master/src/GameConnection/config.toml) file.

## Running the project

To run the project the user must:
+ [Configure the project;](https://github.com/Simbs38/tese/tree/master/src)
+ [Install all dependencys;](https://github.com/Simbs38/tese#dependencys)
+ Open the webtiles game in the browser and enter the game;
+ Run the make command from the main folder;
+ Click the webtiles game to make the commands afect the game.

## Dependencys

+ PyTorch

## The MakeFile

To run the network, run the command make, to analyse the results, run the comand make res, after inserting all of the results in the folder results

## Project Arquitecture

The project consists of several packages that work together. 

### ReinforcementBot

There are 4 main packages in the project:

+ [Reinforcement](https://github.com/Simbs38/tese/tree/master/src/ReinforcementBot/Reinforcement)
+ [Environment](https://github.com/Simbs38/tese/tree/master/src/ReinforcementBot/Environment)
+ [GameConnection](https://github.com/Simbs38/tese/tree/master/src/ReinforcementBot/GameConnection)
+ [WebtilesHanlder](https://github.com/Simbs38/tese/tree/master/src/ReinforcementBot/WebtilesHandler)

## Project Background

The GameConnection package is simplification of the [Beem project](https://github.com/gammafunk/beem), where we just use the code that gets us the game information as 


This project was made in colaboration with [Universidade do Minho](https://www.uminho.pt/PT) in the [Informatics Department](https://www.di.uminho.pt/) as a Masters Degrees Thesis in the area of Artificial Inteligence.
