install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:	
	black *.py 

# lint:
# 	pylint --disable=R,C --ignore-patterns=test_.*?py *.py src/*.py

container-lint:
	docker run --rm -i hadolint/hadolint < Dockerfile

refactor: format

deploy:
	#deploy goes here
		
all: instal test format deploy
