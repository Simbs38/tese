

run: 
	python3 beem/server.py

serve:
	python3 server/StateUpdate.py

clean:
	sudo rm -rf beem/__pycache__
	sudo rm -rf server/__pycache__