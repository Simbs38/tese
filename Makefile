

run:
	clear
	python3 src/ReinforcementBot/main.py

old:
	while true  ; do \
		python3 src/ReinforcementBot/main.py ; \
	done

dev:
	clear
	python3 src/ReinforcementBot/mainDev.py

clean:
	echo "To Do"