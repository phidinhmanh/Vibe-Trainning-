.PHONY: all setup clean train evaluate test lint cluster

# Configuration
CONFIG ?= configs/default.yaml
DATA_DIR ?= data
EXPERIMENTS_DIR ?= experiments
SEED ?= 42

# Python
PYTHON ?= python3
PIP ?= pip

# =============================================================================
# Setup
# =============================================================================

setup:
	$(PIP) install -r requirements.txt
	mkdir -p $(DATA_DIR)/raw $(DATA_DIR)/processed $(EXPERIMENTS_DIR)
	mkdir -p resources notebooks

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black flake8 mypy

# =============================================================================
# Data Processing
# =============================================================================

download-data:
	@echo "Please download data manually from Kaggle:"
	@echo "  1. Train sequences: Train/train_sequences.fasta"
	@echo "  2. Train terms: Train/train_terms.tsv"
	@echo "  3. Test sequences: Test/testsuperset.fasta"
	@echo "Place files in data/raw/"

cluster:
	@echo "Clustering sequences for homology-aware splits..."
	$(PYTHON) -c "from src.data.homology_split import HomologySplitter; \
		from src.data.ingest import load_sequences; \
		seqs = load_sequences('$(DATA_DIR)/raw/train_sequences.fasta'); \
		splitter = HomologySplitter(n_folds=5, identity_threshold=0.7); \
		splitter.fit(seqs); \
		splitter.save('$(DATA_DIR)/folds.json'); \
		print(f'Created {splitter.n_folds} folds with {len(splitter.clusters_)} clusters')"

# =============================================================================
# Training
# =============================================================================

train-baseline:
	@echo "Training baseline LR model..."
	$(PYTHON) -m src.models.baseline_lr --config $(CONFIG)

train-knn:
	@echo "Training KNN transfer model..."
	$(PYTHON) -m src.models.knn_transfer --config $(CONFIG)

train-all: train-baseline train-knn
	@echo "All base models trained."

# =============================================================================
# Optimization
# =============================================================================

optuna:
	@echo "Running Optuna hyperparameter optimization..."
	$(PYTHON) src/models/optuna_search.py --config $(CONFIG) --n-trials 100

# =============================================================================
# Evaluation
# =============================================================================

evaluate:
	@echo "Evaluating model..."
	$(PYTHON) -c "from src.score.scorer import CafaScorer; print('Scorer loaded successfully')"

score-submission:
	@echo "Scoring submission file..."
	$(PYTHON) src/score/scorer.py --submission submission.tsv --ground-truth $(DATA_DIR)/raw/train_terms.tsv

# =============================================================================
# Inference
# =============================================================================

infer:
	$(PYTHON) src/models/infer.py --config $(CONFIG) --input test.fasta --output predictions.npy

submit:
	$(PYTHON) src/models/infer.py --config $(CONFIG) --input $(DATA_DIR)/raw/testsuperset.fasta --output submission.tsv --format tsv

# =============================================================================
# Testing & Quality
# =============================================================================

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src/ tests/ --max-line-length=100
	black --check src/ tests/

format:
	black src/ tests/

typecheck:
	mypy src/ --ignore-missing-imports

# =============================================================================
# Utilities
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov/

clean-experiments:
	rm -rf $(EXPERIMENTS_DIR)/*

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker build -t cafa6:latest .

docker-run:
	docker run --rm -v $(PWD)/data:/app/data -v $(PWD)/experiments:/app/experiments cafa6:latest

# =============================================================================
# Help
# =============================================================================

help:
	@echo "CAFA 6 Protein Function Prediction"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Setup:"
	@echo "  setup           Install dependencies"
	@echo "  install-dev     Install dev dependencies (pytest, lint)"
	@echo ""
	@echo "Data:"
	@echo "  download-data   Instructions for downloading data"
	@echo "  cluster         Generate homology-aware splits"
	@echo ""
	@echo "Training:"
	@echo "  train-baseline  Train baseline LR model"
	@echo "  train-knn       Train KNN transfer model"
	@echo "  train-all       Train all base models"
	@echo "  optuna          Run hyperparameter optimization"
	@echo ""
	@echo "Inference:"
	@echo "  infer           Run inference on test set"
	@echo "  submit          Generate submission file"
	@echo ""
	@echo "Quality:"
	@echo "  test            Run tests"
	@echo "  lint            Check code style"
	@echo "  format          Auto-format code"
	@echo ""
	@echo "Utilities:"
	@echo "  clean           Remove cache files"
	@echo "  help            Show this message"
