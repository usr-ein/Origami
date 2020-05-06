init:
	pip3 install --user -r requirements.txt

test:
	mypy mlmodel --ignore-missing-imports
	nosetests tests