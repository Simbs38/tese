
run:
	python3 src/ReinforcementBot/main.py & python3 src/GameConnection/main.py

bot:
	python3 src/ReinforcementBot/main.py

server: 
	python3 src/GameConnection/main.py

dev:
	clear
	python3 src/ReinforcementBot/Environment/main.py


clean:
	echo "To Do"