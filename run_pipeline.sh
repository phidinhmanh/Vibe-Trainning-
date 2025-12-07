#!/bin/bash
# =============================================================================
# CAFA 6 Complete Pipeline Runner
# =============================================================================
# This script runs the entire CAFA 6 pipeline from setup to submission.
#
# Usage:
#   ./run_pipeline.sh              # Run full pipeline
#   ./run_pipeline.sh --skip-setup # Skip setup if already installed
#   ./run_pipeline.sh --test-only  # Run only tests
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONFIG="configs/default.yaml"
OPTUNA_TRIALS=100
SEED=42

# Logging
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Timer
start_timer() { START_TIME=$(date +%s); }
end_timer() {
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo -e "${GREEN}Completed in ${ELAPSED}s${NC}"
}

# Parse arguments
SKIP_SETUP=false
TEST_ONLY=false

for arg in "$@"; do
    case $arg in
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --test-only)
            TEST_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-setup    Skip pip install (if already done)"
            echo "  --test-only     Run only tests, skip training"
            echo "  --help          Show this help message"
            exit 0
            ;;
    esac
done

# =============================================================================
# Main Pipeline
# =============================================================================

echo ""
echo "=============================================="
echo "  CAFA 6 Protein Function Prediction"
echo "  Complete Pipeline Runner"
echo "=============================================="
echo ""

# Check if we're in the right directory
if [ ! -f "Makefile" ]; then
    log_error "Makefile not found. Please run this script from the cafa6 directory."
    exit 1
fi

# Check for data files
check_data() {
    log_info "Checking data files..."
    
    # Robust file finding using 'find'
    # We search in ../kaggle/input first, then data/raw
    
    find_file() {
        local name=$1
        local found=""
        
        # Check Kaggle first (Absolute path)
        if [ -d "/kaggle/input" ]; then
            found=$(find /kaggle/input -name "$name" | head -n 1)
        fi

        # Check relative ../input or ../kaggle/input
        if [ -z "$found" ] && [ -d "../input" ]; then
            found=$(find ../input -name "$name" | head -n 1)
        fi
        
        # Check local if not found
        if [ -z "$found" ] && [ -d "data/raw" ]; then
            found=$(find data/raw -name "$name" | head -n 1)
        fi
        
        echo "$found"
    }
    
    TRAIN_SEQ=$(find_file "train_sequences.fasta")
    TRAIN_TERMS=$(find_file "train_terms.tsv")
    TEST_SEQ=$(find_file "testsuperset.fasta")
    
    REQ_FILES=("$TRAIN_SEQ" "$TRAIN_TERMS" "$TEST_SEQ")
    MISSING=false
    
    if [ -z "$TRAIN_SEQ" ]; then log_error "Missing: train_sequences.fasta"; MISSING=true; else log_info "Found: $TRAIN_SEQ"; fi
    if [ -z "$TRAIN_TERMS" ]; then log_error "Missing: train_terms.tsv"; MISSING=true; else log_info "Found: $TRAIN_TERMS"; fi
    if [ -z "$TEST_SEQ" ]; then log_error "Missing: testsuperset.fasta"; MISSING=true; else log_info "Found: $TEST_SEQ"; fi
    
    if [ "$MISSING" = true ]; then
        log_error "Critical data files not found in ../kaggle/input or data/raw"
        exit 1
    fi
    
    log_success "All required data files found!"
}

# Step 1: Setup with virtual environment
step_setup() {
    if [ "$SKIP_SETUP" = true ]; then
        log_warning "Skipping setup (--skip-setup flag)"
        # Still need to activate venv if it exists
        if [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
        fi
        return
    fi
    
    log_info "Step 1: Setting up virtual environment and installing dependencies..."
    start_timer
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip --quiet
    
    # Install dependencies
    pip install -r requirements.txt --quiet
    
    end_timer
    log_success "Virtual environment created and dependencies installed!"
    log_info "Virtual environment: $(which python)"
}

# Step 2: Run tests
step_test() {
    log_info "Step 2: Running unit tests..."
    start_timer
    
    python -m pytest tests/ -v --tb=short || {
        log_error "Tests failed!"
        exit 1
    }
    
    end_timer
    log_success "All tests passed!"
}

# Step 3: Create homology splits (FAST with MinHash LSH)
step_cluster() {
    log_info "Step 3: Creating homology-aware splits (using MinHash LSH)..."
    start_timer
    
    if [ -f "data/folds.json" ] && [ -s "data/folds.json" ]; then
        # Verify the file has clusters key
        if python -c "import json; d=json.load(open('data/folds.json')); assert 'clusters' in d" 2>/dev/null; then
            log_warning "data/folds.json already exists. Skipping clustering."
            log_info "Delete data/folds.json to regenerate."
            end_timer
            return
        else
            log_warning "data/folds.json is invalid, regenerating..."
            rm -f data/folds.json
        fi
    fi
    
    python -c "
from src.data.fast_cluster import MinHashLSH
from src.data.ingest import load_sequences
from collections import defaultdict
import numpy as np
import json

print('Loading sequences...')
from src.utils import load_config
config = load_config('configs/default.yaml')
seqs = load_sequences(config['data']['train_sequences'])
print(f'Loaded {len(seqs)} sequences')

print('Clustering with MinHash LSH (fast approximate clustering)...')
lsh = MinHashLSH(num_hashes=128, bands=16, k=3)
clusters = lsh.cluster(seqs, min_similarity=0.7)

print(f'Created {len(clusters)} clusters')

# Assign clusters to folds (balanced by size)
n_folds = 5
cluster_ids = list(clusters.keys())
cluster_sizes = {cid: len(clusters[cid]) for cid in cluster_ids}

# Sort by size (largest first) and assign to smallest fold
np.random.seed(42)
np.random.shuffle(cluster_ids)
cluster_ids = sorted(cluster_ids, key=lambda x: -cluster_sizes[x])

fold_sizes = [0] * n_folds
fold_assignments = [[] for _ in range(n_folds)]

for cid in cluster_ids:
    min_fold = np.argmin(fold_sizes)
    fold_assignments[min_fold].append(cid)
    fold_sizes[min_fold] += cluster_sizes[cid]

print(f'Fold sizes: {fold_sizes}')

# Save
data = {
    'description': 'Homology-aware k-fold splits (MinHash LSH)',
    'n_folds': n_folds,
    'identity_threshold': 0.7,
    'clustering_method': 'minhash_lsh',
    'folds': fold_assignments,
    'clusters': clusters,
    'metadata': {
        'total_clusters': len(clusters),
        'total_proteins': sum(len(v) for v in clusters.values())
    }
}

with open('data/folds.json', 'w') as f:
    json.dump(data, f)

print(f'Saved to data/folds.json')
"
    
    end_timer
    log_success "Homology splits created!"
}

# Step 4: Train baseline model
step_train_baseline() {
    log_info "Step 4: Training baseline LR model..."
    start_timer
    
    python -c "
import numpy as np
import json
from src.data.ingest import load_sequences, load_terms, get_top_go_terms, prepare_labels
from src.data.homology_split import HomologySplitter
from src.features.tfidf_kmers import TfidfKmerFeaturizer
from src.models.baseline_lr import BaselineLR
from src.score.scorer import CafaScorer
from src.utils import set_seed
import os

set_seed(42)

print('Loading data...')
from src.utils import load_config
config = load_config('configs/default.yaml')
sequences = load_sequences(config['data']['train_sequences'])
terms_df = load_terms(config['data']['train_terms'])
protein_ids = list(sequences.keys())

print('Preparing labels...')
top_go = get_top_go_terms(terms_df, n_terms=1500)
y, go2idx = prepare_labels(protein_ids, terms_df, top_go)

print('Extracting TF-IDF features...')
seq_list = [sequences[pid] for pid in protein_ids]
featurizer = TfidfKmerFeaturizer(k=3, max_features=200000)
X = featurizer.fit_transform(seq_list)
print(f'Features shape: {X.shape}')

print('Loading splits...')
splitter = HomologySplitter.load('data/folds.json')
pid_to_idx = {pid: i for i, pid in enumerate(protein_ids)}

# Train on folds 0-3, validate on fold 4
val_fold = 4
train_ids, val_ids = splitter.get_fold(val_fold)
train_idx = [pid_to_idx[p] for p in train_ids if p in pid_to_idx]
val_idx = [pid_to_idx[p] for p in val_ids if p in pid_to_idx]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

print(f'Train: {len(train_idx)} samples, Val: {len(val_idx)} samples')

print('Training LR model...')
model = BaselineLR(C=1.0, calibrate=True, max_iter=1000)

# Split train for calibration
n_train = int(0.8 * X_train.shape[0])
model.fit(X_train[:n_train], y_train[:n_train], X_train[n_train:], y_train[n_train:])

print('Evaluating...')
proba = model.predict_proba(X_val)
scorer = CafaScorer()
result = scorer.score(y_val, proba)

print(f'\\nResults:')
print(f'  F-Max: {result.f1:.4f}')
print(f'  Best Threshold: {result.tau:.4f}')
print(f'  Weighted Precision: {result.weighted_precision:.4f}')
print(f'  Weighted Recall: {result.weighted_recall:.4f}')

# Save model and results
os.makedirs('experiments/models', exist_ok=True)
model.save('experiments/models/baseline_lr.joblib')
featurizer.save('experiments/models/tfidf_vectorizer.joblib')

with open('experiments/models/go_terms.json', 'w') as f:
    json.dump(top_go, f)

with open('experiments/baseline_results.json', 'w') as f:
    json.dump({
        'f_max': result.f1,
        'threshold': result.tau,
        'precision': result.weighted_precision,
        'recall': result.weighted_recall
    }, f, indent=2)

print('\\nModel saved to experiments/models/')
"
    
    end_timer
    log_success "Baseline model trained!"
}

# Step 5: Train KNN model
step_train_knn() {
    log_info "Step 5: Training KNN transfer model..."
    start_timer
    
    python -c "
import numpy as np
import json
from src.data.ingest import load_sequences, load_terms, get_top_go_terms, prepare_labels
from src.data.homology_split import HomologySplitter
from src.features.physchem import extract_features
from src.models.knn_transfer import KNNTransfer
from src.score.scorer import CafaScorer
from src.utils import set_seed
import os

set_seed(42)

print('Loading data...')
from src.utils import load_config
config = load_config('configs/default.yaml')
sequences = load_sequences(config['data']['train_sequences'])
terms_df = load_terms(config['data']['train_terms'])
protein_ids = list(sequences.keys())

print('Preparing labels...')
top_go = get_top_go_terms(terms_df, n_terms=1500)
y, go2idx = prepare_labels(protein_ids, terms_df, top_go)

print('Extracting physicochemical features...')
seq_list = [sequences[pid] for pid in protein_ids]
X = extract_features(seq_list, n_features=85)
print(f'Features shape: {X.shape}')

print('Loading splits...')
splitter = HomologySplitter.load('data/folds.json')
pid_to_idx = {pid: i for i, pid in enumerate(protein_ids)}

val_fold = 4
train_ids, val_ids = splitter.get_fold(val_fold)
train_idx = [pid_to_idx[p] for p in train_ids if p in pid_to_idx]
val_idx = [pid_to_idx[p] for p in val_ids if p in pid_to_idx]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

print(f'Train: {len(train_idx)} samples, Val: {len(val_idx)} samples')

print('Training KNN model...')
model = KNNTransfer(n_neighbors=7, metric='cosine')
model.fit(X_train, y_train, [protein_ids[i] for i in train_idx])

print('Evaluating...')
proba = model.predict_proba(X_val)
scorer = CafaScorer()
result = scorer.score(y_val, proba)

print(f'\\nResults:')
print(f'  F-Max: {result.f1:.4f}')
print(f'  Best Threshold: {result.tau:.4f}')
print(f'  Weighted Precision: {result.weighted_precision:.4f}')
print(f'  Weighted Recall: {result.weighted_recall:.4f}')

# Save model
os.makedirs('experiments/models', exist_ok=True)
model.save('experiments/models/knn_transfer.joblib')

with open('experiments/knn_results.json', 'w') as f:
    json.dump({
        'f_max': result.f1,
        'threshold': result.tau,
        'precision': result.weighted_precision,
        'recall': result.weighted_recall
    }, f, indent=2)

print('\\nModel saved to experiments/models/')
"
    
    end_timer
    log_success "KNN model trained!"
}

# Step 6: Generate submission
step_submit() {
    log_info "Step 6: Generating submission file..."
    start_timer
    
    python -c "
import numpy as np
import json
from src.data.ingest import load_sequences
from src.features.tfidf_kmers import TfidfKmerFeaturizer
from src.features.physchem import extract_features
from src.models.baseline_lr import BaselineLR
from src.models.knn_transfer import KNNTransfer
import os

print('Loading test sequences...')
from src.utils import load_config
config = load_config('configs/default.yaml')
test_seqs = load_sequences(config['data']['test_sequences'], parse_uniprot=False)
test_ids = list(test_seqs.keys())
test_list = [test_seqs[pid] for pid in test_ids]
print(f'Loaded {len(test_seqs)} test sequences')

print('Loading models...')
lr_model = BaselineLR.load('experiments/models/baseline_lr.joblib')
knn_model = KNNTransfer.load('experiments/models/knn_transfer.joblib')

print('Loading featurizer...')
featurizer = TfidfKmerFeaturizer()
featurizer.load('experiments/models/tfidf_vectorizer.joblib')

print('Loading GO terms...')
with open('experiments/models/go_terms.json', 'r') as f:
    go_terms = json.load(f)

print('Extracting features...')
X_tfidf = featurizer.transform(test_list)
X_phys = extract_features(test_list, n_features=85)

print('Generating predictions...')
pred_lr = lr_model.predict_proba(X_tfidf)
pred_knn = knn_model.predict_proba(X_phys)

# Ensemble (weighted average)
knn_weight = 0.32
lr_weight = 1.0 - knn_weight
combined = lr_weight * pred_lr + knn_weight * pred_knn

print('Writing submission...')
with open('submission.tsv', 'w') as f:
    for i, pid in enumerate(test_ids):
        scores = combined[i]
        sorted_idx = np.argsort(-scores)[:1500]
        for idx in sorted_idx:
            if scores[idx] >= 0.01:
                f.write(f'{pid}\t{go_terms[idx]}\t{scores[idx]:.4f}\n')

# Count predictions
with open('submission.tsv', 'r') as f:
    n_preds = sum(1 for _ in f)

print(f'\\nSubmission saved: submission.tsv')
print(f'Total predictions: {n_preds:,}')
print(f'Average per protein: {n_preds/len(test_ids):.1f}')
"
    
    end_timer
    log_success "Submission file generated!"
}

# Step 7: Summary
step_summary() {
    echo ""
    echo "=============================================="
    echo "  Pipeline Complete!"
    echo "=============================================="
    echo ""
    
    if [ -f "experiments/baseline_results.json" ]; then
        echo "Baseline LR Results:"
        cat experiments/baseline_results.json
        echo ""
    fi
    
    if [ -f "experiments/knn_results.json" ]; then
        echo "KNN Results:"
        cat experiments/knn_results.json
        echo ""
    fi
    
    if [ -f "submission.tsv" ]; then
        echo "Submission: submission.tsv"
        echo "  Lines: $(wc -l < submission.tsv)"
        echo "  Size: $(du -h submission.tsv | cut -f1)"
    fi
    
    echo ""
    log_success "All steps completed successfully!"
}

# =============================================================================
# Run Pipeline
# =============================================================================

check_data

if [ "$TEST_ONLY" = true ]; then
    step_setup
    step_test
    log_success "Test-only mode complete!"
    exit 0
fi

TOTAL_START=$(date +%s)

step_setup
step_test
step_cluster
step_train_baseline
step_train_knn
step_submit
step_summary

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
echo ""
log_success "Total pipeline time: ${TOTAL_ELAPSED}s ($(($TOTAL_ELAPSED / 60))m $(($TOTAL_ELAPSED % 60))s)"
