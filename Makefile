

run: 
	python3 setup.py build
	sudo python3 setup.py install
	beem

clean:
	sudo rm -rf beem.egg-info
	sudo rm -rf build
	sudo rm -rf dist