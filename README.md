# CAFA 6 Protein Function Prediction

A production-ready pipeline for the [CAFA 6 Kaggle competition](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction).

## Features

- **Homology-aware data splitting**: Prevents data leakage from similar sequences
- **Multiple feature extraction**: TF-IDF k-mers, physicochemical properties
- **Model ensemble**: Logistic Regression, KNN label transfer, stacking
- **CAFA-compliant scoring**: Weighted F-Max with Information Accretion (IA) weights
- **Hyperparameter optimization**: Optuna integration with F-Max objective
- **GO hierarchy enforcement**: True-Path Rule propagation

## Project Structure

```
cafa6/
├── data/                  # Raw, processed data, fold assignments
│   ├── raw/
│   ├── processed/
│   └── folds.json
├── configs/
│   └── default.yaml       # Configuration file
├── src/
│   ├── data/              # Data loading and splitting
│   │   ├── ingest.py
│   │   └── homology_split.py
│   ├── features/          # Feature extraction
│   │   ├── tfidf_kmers.py
│   │   └── physchem.py
│   ├── models/            # ML models
│   │   ├── baseline_lr.py
│   │   ├── knn_transfer.py
│   │   ├── stacker.py
│   │   ├── optuna_search.py
│   │   └── infer.py
│   ├── score/             # Evaluation metrics
│   │   ├── scorer.py
│   │   ├── thresholds.py
│   │   └── hierarchy.py
│   └── utils/
├── tests/                 # Unit tests
├── notebooks/             # Exploration notebooks
├── experiments/           # Saved models, logs
├── Makefile
├── requirements.txt
└── Dockerfile
```

## Quick Start

### 1. Setup

```bash
# Clone and install
cd cafa6
pip install -r requirements.txt

# Or use make
make setup
```

### 2. Prepare Data

Download data from Kaggle and place in `data/raw/`:
- `train_sequences.fasta`
- `train_terms.tsv`
- `testsuperset.fasta`

### 3. Create Homology-Aware Splits

```bash
make cluster
```

This creates `data/folds.json` with cluster-aware fold assignments.

### 4. Train Baseline Model

```python
from src.data.ingest import load_sequences, load_terms, get_top_go_terms, prepare_labels
from src.data.homology_split import HomologySplitter
from src.features.tfidf_kmers import TfidfKmerFeaturizer
from src.models.baseline_lr import BaselineLR
from src.score.scorer import CafaScorer

# Load data
sequences = load_sequences('data/raw/train_sequences.fasta')
terms_df = load_terms('data/raw/train_terms.tsv')
protein_ids = list(sequences.keys())

# Prepare labels
top_go = get_top_go_terms(terms_df, n_terms=1500)
y, go2idx = prepare_labels(protein_ids, terms_df, top_go)

# Extract features
seq_list = [sequences[pid] for pid in protein_ids]
featurizer = TfidfKmerFeaturizer(k=3, max_features=200000)
X = featurizer.fit_transform(seq_list)

# Split data (homology-aware)
splitter = HomologySplitter.load('data/folds.json')
train_ids, val_ids = splitter.get_fold(0)

# Train
model = BaselineLR(C=1.0, calibrate=True)
model.fit(X_train, y_train)

# Evaluate
proba = model.predict_proba(X_val)
scorer = CafaScorer()
result = scorer.score(y_val, proba)
print(f"F-Max: {result.f1:.4f} at threshold {result.tau:.3f}")
```

### 5. Hyperparameter Optimization

```bash
python src/models/optuna_search.py --config configs/default.yaml --n-trials 100
```

### 6. Generate Submission

```bash
python src/models/infer.py \
    --config configs/default.yaml \
    --input data/raw/testsuperset.fasta \
    --output submission.tsv \
    --format tsv
```

## Key Concepts

### Homology-Aware Splitting

**Problem**: Random splits leak information when similar sequences appear in both train and validation sets.

**Solution**: Cluster sequences by similarity (MMseqs2/CD-HIT/k-mer Jaccard), then assign entire clusters to folds.

```python
splitter = HomologySplitter(
    n_folds=5,
    identity_threshold=0.7,  # Max identity between folds
    method='mmseqs2'         # or 'cdhit', 'kmer'
)
splitter.fit(sequences)
```

### CAFA Scoring

The official metric is weighted F-Max:

```python
from src.score.scorer import weighted_precision_recall_f1

result = weighted_precision_recall_f1(
    y_true,        # (N, C) binary
    y_prob,        # (N, C) probabilities
    ia_weights,    # (C,) Information Accretion weights
    thresholds     # Array of thresholds to sweep
)
# Returns: {'tau': best_threshold, 'f1': f_max, 'wprec': ..., 'wrec': ...}
```

### Threshold Optimization

Global threshold sweep is mandatory. Per-label thresholds can improve further:

```python
from src.score.thresholds import ThresholdOptimizer

optimizer = ThresholdOptimizer(strategy='per_label', top_k_per_label=100)
optimizer.fit(y_val, proba_val)
predictions = optimizer.predict(proba_test)
```

### GO Hierarchy Enforcement

Apply True-Path Rule AFTER calibration, BEFORE thresholding:

```python
from src.score.hierarchy import GOHierarchy

hierarchy = GOHierarchy(obo_path='resources/go.obo')
propagated_scores = hierarchy.propagate_max(scores, term_to_idx)
```

## Ensemble Strategy

1. **Level 0 (Base Models)**:
   - Logistic Regression on TF-IDF k-mers
   - KNN label transfer on physicochemical features
   - (Optional) XGBoost, MLP on embeddings

2. **Level 1 (Meta-Learner)**:
   - Train on out-of-fold predictions from same GroupKFold
   - Ridge/Logistic regression per class

```python
from src.models.stacker import Stacker

stacker = Stacker(meta_learner='ridge', per_class_meta=True)
stacker.fit(oof_predictions=[oof_lr, oof_knn], y_true=y_train)
final_pred = stacker.predict_proba([pred_lr, pred_knn])
```

## Testing

```bash
make test
# or
pytest tests/ -v
```

## Common Pitfalls

1. ❌ **Random split** → ✅ Homology-aware split
2. ❌ **Optimize standard loss** → ✅ Optimize weighted F-Max
3. ❌ **Skip label transfer** → ✅ KNN/BLAST for rare labels
4. ❌ **Tune thresholds on test** → ✅ Use validation fold only
5. ❌ **Propagate before calibration** → ✅ Calibrate first, then propagate

## License

MIT
