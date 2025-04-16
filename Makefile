# HMER-IM2LATEX Makefile

# Set default variables
PYTHON := python
PIP := pip
CONFIG := img2latex/configs/default.yaml
MODEL := outputs/cnn_lstm_256/checkpoints/best_model.pt
IMAGE := data/test_images/sample.png
EXPERIMENT := cnn_lstm
CHECKPOINTS_DIR := outputs/checkpoints
OUTPUTS_DIR := outputs
TEST_IMAGES_DIR := data/test_images

# Clean targets
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '.pytest_cache' -exec rm -rf {} +
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.egg' -exec rm -f {} +

clean-outputs:
	@echo "Removing all model outputs..."
	rm -rf $(CHECKPOINTS_DIR)/*
	rm -rf $(OUTPUTS_DIR)/*
	@echo "Outputs directory completely cleaned"

clean-all: clean-pyc clean-outputs
	@echo "All cleaning operations completed"

# Setup targets
setup:
	@echo "Installing dependencies..."
	$(PIP) install torch torchvision
	$(PIP) install -e .
	@echo "Dependencies installed successfully"

# Create necessary directories
dirs:
	@mkdir -p $(CHECKPOINTS_DIR)
	@mkdir -p $(OUTPUTS_DIR)
	@mkdir -p $(TEST_IMAGES_DIR)
	@echo "Directories created"

# Download dataset
download-data:
	@echo "Downloading IM2LATEX dataset..."
	$(PYTHON) -c "import kagglehub; kagglehub.dataset_download('shahrukhkhan/im2latex100k')"
	@echo "Dataset downloaded successfully"

# Training targets
train:
	@echo "Starting training using config: $(CONFIG)"
	$(PYTHON) -m prime_vit train --config $(CONFIG)

# Prediction targets
predict:
	@echo "Running prediction on image: $(IMAGE) with model: $(MODEL)"
	$(PYTHON) -m prime_vit predict --model $(MODEL) --config $(CONFIG) --image $(IMAGE)

# Training with experiment versioning
train-exp:
	@if [ -z "$(EXPERIMENT)" ]; then \
		BASE_NAME="im2latex"; \
	else \
		BASE_NAME="$(EXPERIMENT)"; \
	fi; \
	VERSION=1; \
	while [ -d "$(OUTPUTS_DIR)/$${BASE_NAME}_v$${VERSION}" ]; do \
		VERSION=$$((VERSION + 1)); \
	done; \
	EXP_NAME="$${BASE_NAME}_v$${VERSION}"; \
	echo "Starting training for experiment: $$EXP_NAME"; \
	mkdir -p $(OUTPUTS_DIR)/$$EXP_NAME; \
	cp $(CONFIG) $(OUTPUTS_DIR)/$$EXP_NAME/; \
	$(PYTHON) -m prime_vit train --config $(CONFIG) --output $(OUTPUTS_DIR)/$$EXP_NAME | tee $(OUTPUTS_DIR)/$$EXP_NAME/train.log

# Evaluation on test set
evaluate:
	@echo "Evaluating model $(MODEL) on test set"
	$(PYTHON) -m prime_vit evaluate --model $(MODEL) --config $(CONFIG)

# Code quality targets
lint:
	ruff check .

lint-fix:
	ruff check --fix .

format:
	ruff format .

typecheck:
	mypy .

check-all: lint format typecheck

# Help target
help:
	@echo "HMER-IM2LATEX Makefile targets:"
	@echo "  setup             - Install dependencies"
	@echo "  dirs              - Create necessary directories"
	@echo "  download-data     - Download the IM2LATEX dataset"
	@echo "  train             - Train the model (use CONFIG=path to customize)"
	@echo "  train-exp         - Train with experiment versioning (use EXPERIMENT=name)"
	@echo "  predict           - Run prediction on an image (use MODEL=path IMAGE=path)"
	@echo "  evaluate          - Evaluate model on test set (use MODEL=path)"
	@echo "  clean-pyc         - Remove Python file artifacts"
	@echo "  clean-outputs     - Remove all model outputs"
	@echo "  clean-all         - Run all cleaning targets"
	@echo "  lint              - Check code with linter"
	@echo "  lint-fix          - Fix linting issues automatically"
	@echo "  format            - Format code according to style guide"
	@echo "  typecheck         - Run type checking"
	@echo "  check-all         - Run all code quality checks"
	@echo "  help              - Show this help message"

.PHONY: clean-pyc clean-outputs clean-all setup dirs download-data train train-exp predict evaluate lint lint-fix format typecheck check-all help 