init:
	pip install -r requirements.txt

test:
	mypy mlmodel --ignore-missing-imports
	nosetests tests