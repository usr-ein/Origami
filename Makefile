init:
	pip3 install --user -r requirements.txt

test:
	mypy origami --ignore-missing-imports
	nosetests tests