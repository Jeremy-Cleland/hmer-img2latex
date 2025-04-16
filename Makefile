# HMER-IM2LATEX Makefile

# Set default variables
PYTHON := python
PIP := pip
CONFIG := img2latex/configs/config.yaml
MODEL := outputs/img2latex_v1/checkpoints/best_checkpoint.pt
IMAGE := data/img
EXPERIMENT := img2latex
DATA_DIR := data
CHECKPOINTS_DIR := outputs/checkpoints
OUTPUTS_DIR := outputs
TEST_IMAGES_DIR := data/test_images
METRICS_DIR := outputs/metrics
EPOCHS := 5

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
	rm -rf $(OUTPUTS_DIR)/*
	@echo "Outputs directory completely cleaned"

clean-metrics:
	@echo "Removing metrics files..."
	find $(OUTPUTS_DIR) -path "*/metrics/*" -type f -delete
	@echo "Metrics files cleaned"

clean-all: clean-pyc clean-outputs
	@echo "All cleaning operations completed"

# Setup targets
setup:
	@echo "Installing dependencies..."
	$(PIP) install -e .
	@echo "Dependencies installed successfully"

# Create necessary directories
dirs:
	@mkdir -p $(OUTPUTS_DIR)
	@mkdir -p $(TEST_IMAGES_DIR)
	@mkdir -p $(METRICS_DIR)
	@echo "Directories created"

# Download dataset
# Training targets
train:
	@if [ -z "$(EXPERIMENT)" ]; then \
		BASE_NAME="img2latex"; \
	else \
		BASE_NAME="$(EXPERIMENT)"; \
	fi; \
	if [[ "$$BASE_NAME" == *_v* ]]; then \
		echo "Error: Please provide a base experiment name without version suffix (_v1, _v2, etc.)"; \
		exit 1; \
	fi; \
	VERSION=1; \
	while [ -d "$(OUTPUTS_DIR)/$${BASE_NAME}_v$${VERSION}" ]; do \
		VERSION=$$((VERSION + 1)); \
	done; \
	EXPERIMENT_NAME="$${BASE_NAME}_v$${VERSION}"; \
	echo "Starting training for experiment: $$EXPERIMENT_NAME"; \
	if [ -z "$(CONFIG)" ]; then \
		CONFIG_PATH="img2latex/configs/config.yaml"; \
	else \
		CONFIG_PATH="$(CONFIG)"; \
	fi; \
	$(PYTHON) -m img2latex.cli train --config-path $$CONFIG_PATH --experiment-name $$EXPERIMENT_NAME

train-resume:
	@echo "Resuming training from checkpoint: $(MODEL)"
	$(PYTHON) -m img2latex.cli train --config-path $(CONFIG) --experiment-name $(EXPERIMENT) --checkpoint-path $(MODEL)

# Prediction targets
predict:
	@echo "Running prediction on image: $(IMAGE) with model: $(MODEL)"
	$(PYTHON) -m img2latex.cli predict $(MODEL) $(IMAGE)

# Evaluation on test set
evaluate:
	@echo "Evaluating model $(MODEL) on test set"
	$(PYTHON) -m img2latex.cli evaluate $(MODEL) $(DATA_DIR) --split test

# New metrics targets
metrics-visualize:
	@echo "Visualizing metrics for experiment: $(EXPERIMENT)"
	$(PYTHON) -m img2latex.cli analyze metrics visualize --experiment $(EXPERIMENT)

metrics-latest:
	@echo "Showing latest metrics for experiment: $(EXPERIMENT)"
	$(PYTHON) -m img2latex.cli analyze metrics latest --experiment $(EXPERIMENT)

metrics-compare:
	@echo "Comparing metrics across experiments"
	$(PYTHON) -m img2latex.cli analyze metrics compare

metrics-export:
	@echo "Exporting metrics to CSV for experiment: $(EXPERIMENT)"
	$(PYTHON) -m img2latex.cli analyze metrics export --experiment $(EXPERIMENT) --format csv

# Code quality targets
lint:
	ruff check img2latex

lint-fix:
	ruff check --fix img2latex

format:
	ruff format img2latex

typecheck:
	mypy img2latex

check-all: lint format typecheck

# Help target
help:
	@echo "HMER-IM2LATEX Makefile targets:"
	@echo "  setup             - Install dependencies"
	@echo "  dirs              - Create necessary directories"
	@echo "  download-data     - Download the IM2LATEX dataset"
	@echo "  train             - Train the model (use CONFIG=path to customize)"
	@echo "  train-resume      - Resume training from a checkpoint"
	@echo "  predict           - Run prediction on an image (use MODEL=path IMAGE=path)"
	@echo "  evaluate          - Evaluate model on test set (use MODEL=path)"
	@echo "  metrics-visualize - Visualize metrics for an experiment"
	@echo "  metrics-latest    - Show latest metrics for an experiment"
	@echo "  metrics-compare   - Compare metrics across experiments"
	@echo "  metrics-export    - Export metrics to CSV format"
	@echo "  clean-pyc         - Remove Python file artifacts"
	@echo "  clean-outputs     - Remove all model outputs"
	@echo "  clean-metrics     - Clean only metrics files"
	@echo "  clean-all         - Run all cleaning targets"
	@echo "  lint              - Check code with linter"
	@echo "  lint-fix          - Fix linting issues automatically"
	@echo "  format            - Format code according to style guide"
	@echo "  typecheck         - Run type checking"
	@echo "  check-all         - Run all code quality checks"
	@echo "  help              - Show this help message"
	
# Analysis targets
analyze-images:
	@echo "Running analysis: images"
	$(PYTHON) -m img2latex.cli analyze images

analyze-project:
	@echo "Running analysis: project"
	$(PYTHON) -m img2latex.cli analyze project

analyze-curves:
	@echo "Running analysis: curves"
	$(PYTHON) -m img2latex.cli analyze curves

analyze-tokens:
	@echo "Running analysis: tokens"
	$(PYTHON) -m img2latex.cli analyze tokens

analyze-errors:
	@echo "Running analysis: errors"
	$(PYTHON) -m img2latex.cli analyze errors

analyze-preprocess:
	@echo "Running analysis: preprocess"
	$(PYTHON) -m img2latex.cli analyze preprocess

analyze-all: analyze-images analyze-project analyze-curves analyze-tokens analyze-errors analyze-preprocess metrics-visualize
	@echo "Running all analysis commands"

.PHONY: clean-pyc clean-outputs clean-metrics clean-all setup dirs download-data train train-resume predict evaluate \
        metrics-visualize metrics-latest metrics-compare metrics-export \
        lint lint-fix format typecheck check-all help analyze-images analyze-project \
        analyze-curves analyze-tokens analyze-errors analyze-preprocess analyze-all