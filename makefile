OS := $(shell uname -s)

ifeq ($(OS),Linux)
	PYTHON = python3
	PYTHON_V = venv/bin/python3
    PIP = venv/bin/pip
    ACTIVATE = venv/bin/activate
else
	PYTHON = python
	PYTHON_V = venv/Scripts/python
	PIP = venv/Scripts/pip
	ACTIVATE = venv/Scripts/activate
endif

.DEFAULT_GOAL := help

.PHONY: extract process train run clean venv help check_config

$(ACTIVATE): requirements.txt
	$(PYTHON) -m venv venv
	chmod +x $(ACTIVATE)
	. $(ACTIVATE)
	$(PIP) install -r requirements.txt

venv: $(ACTIVATE)
	. ./$(ACTIVATE)

# Run extract_frames command
extract: venv
	$(PYTHON_V) main.py extract_frames --actions $(ACTIONS)

# Run process_data command
process: venv
	$(PYTHON_V) main.py process_data

# Run train command
train: venv
	$(PYTHON_V) main.py train

# Run run command
run: venv
	$(PYTHON_V) main.py run

# Remove the virtual environment
clean:
	rm -rf venv

check_config:
	@test -f $(CONFIG) || (echo "Error: $(CONFIG) not found"; exit 1)

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  extract            Extract frames to create a frame dataset"
	@echo "  process            Process the frame dataset and save data as numpy files"
	@echo "  train              Train the model"
	@echo "  run                Run real-time predictions using the trained model"
	@echo "  clean              Cleans up the compiled python artifacts"

extract process train predict: check_config