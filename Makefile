
.PHONY: install-dependencies
# install system dependencies needed to run the environment packages
install-dependencies:
	@echo "Installing Dependencies"
	sudo apt-get update
	sudo apt-get install liblapack-dev libblas-dev gfortran pkg-config libhdf5-dev \
	build-essential libssl-dev zlib1g-dev \
	libbz2-dev libreadline-dev libsqlite3-dev curl \
	libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# custom targets
.PHONY: environment
# setup python environment
environment:
	pyenv install -s 3.9.13 ;\
	pyenv virtualenv 3.9.13 nlp ;\
	pyenv local nlp

.PHONY: requirements
# install core requirements
requirements:
	pip install --upgrade pip
	pip install pdm
	pdm install

.PHONY: jupyter
# start a jupyter notebook server
jupyter:
	PYTHONPATH=$(shell pwd) python -m jupyter notebook