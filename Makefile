

run: 
	python3 GameConnection/server.py

serve:
	python3 server/StateUpdate.py

clean:
	sudo rm -rf GameConnection/__pycache__
	sudo rm -rf server/__pycache__