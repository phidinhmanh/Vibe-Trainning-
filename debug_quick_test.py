#!/usr/bin/env python3
"""
Quick debug script to verify the pipeline works on a small subset.
Tests: data loading, feature extraction, model training, scoring.
"""

import numpy as np
from src.data.ingest import load_sequences, load_terms, get_top_go_terms, prepare_labels
from src.features.tfidf_kmers import TfidfKmerFeaturizer
from src.features.physchem import extract_features
from src.models.baseline_lr import BaselineLR
from src.models.knn_transfer import KNNTransfer
from src.score.scorer import CafaScorer
from src.utils import set_seed, load_config

print("=" * 60)
print("CAFA 6 QUICK DEBUG TEST")
print("=" * 60)

set_seed(42)

# Load config to get robust data paths
config = load_config("configs/default.yaml")
seq_path = config["data"]["train_sequences"]
term_path = config["data"]["train_terms"]

print(f"[INFO] Using sequences from: {seq_path}")
print(f"[INFO] Using terms from: {term_path}")

# 1. Load small subset of data
print("\n[1/6] Loading data (first 500 sequences)...")
sequences = load_sequences(str(seq_path))
terms_df = load_terms(str(term_path))

# Take only first 500 sequences for quick test
protein_ids = list(sequences.keys())[:500]
sequences = {pid: sequences[pid] for pid in protein_ids}
print(f"✓ Loaded {len(sequences)} sequences")

# 2. Prepare labels
print("\n[2/6] Preparing labels (top 50 GO terms)...")
top_go = get_top_go_terms(terms_df, n_terms=50)  # Only 50 terms for speed
y, go2idx = prepare_labels(protein_ids, terms_df, top_go)
print(f"✓ Label matrix shape: {y.shape}")
print(f"✓ Label density: {y.mean():.4f}")

# 3. Extract features
print("\n[3/6] Extracting TF-IDF features...")
seq_list = [sequences[pid] for pid in protein_ids]
featurizer = TfidfKmerFeaturizer(k=3, max_features=5000)  # Reduced features
X_tfidf = featurizer.fit_transform(seq_list)
print(f"✓ TF-IDF shape: {X_tfidf.shape}")

print("\n[4/6] Extracting physicochemical features...")
X_phys = extract_features(seq_list, n_features=85)
print(f"✓ PhysChem shape: {X_phys.shape}")

# 4. Split data
print("\n[5/6] Splitting train/val (80/20)...")
n_train = int(0.8 * len(protein_ids))
train_idx = list(range(n_train))
val_idx = list(range(n_train, len(protein_ids)))

X_train_tfidf, y_train = X_tfidf[train_idx], y[train_idx]
X_val_tfidf, y_val = X_tfidf[val_idx], y[val_idx]

X_train_phys = X_phys[train_idx]
X_val_phys = X_phys[val_idx]

print(f"✓ Train: {len(train_idx)} samples")
print(f"✓ Val: {len(val_idx)} samples")

# 5. Train models
print("\n[6/6] Training models...")

# Logistic Regression
print("  → Training LR (max_iter=100 for speed)...")
lr_model = BaselineLR(C=1.0, calibrate=False, max_iter=100)  # No calibration for speed
lr_model.fit(X_train_tfidf, y_train)
pred_lr = lr_model.predict_proba(X_val_tfidf)
print(f"    ✓ LR predictions shape: {pred_lr.shape}")

# KNN
print("  → Training KNN (k=5)...")
knn_model = KNNTransfer(n_neighbors=5, metric="cosine")
knn_model.fit(X_train_phys, y_train, [protein_ids[i] for i in train_idx])
pred_knn = knn_model.predict_proba(X_val_phys)
print(f"    ✓ KNN predictions shape: {pred_knn.shape}")

# 6. Evaluate
print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

scorer = CafaScorer()

# LR results
result_lr = scorer.score(y_val, pred_lr)
print(f"\nLogistic Regression:")
print(f"  F-Max:     {result_lr.f1:.4f}")
print(f"  Threshold: {result_lr.tau:.4f}")
print(f"  Precision: {result_lr.weighted_precision:.4f}")
print(f"  Recall:    {result_lr.weighted_recall:.4f}")

# KNN results
result_knn = scorer.score(y_val, pred_knn)
print(f"\nKNN Transfer:")
print(f"  F-Max:     {result_knn.f1:.4f}")
print(f"  Threshold: {result_knn.tau:.4f}")
print(f"  Precision: {result_knn.weighted_precision:.4f}")
print(f"  Recall:    {result_knn.weighted_recall:.4f}")

# Ensemble
ensemble = 0.7 * pred_lr + 0.3 * pred_knn
result_ens = scorer.score(y_val, ensemble)
print(f"\nEnsemble (0.7*LR + 0.3*KNN):")
print(f"  F-Max:     {result_ens.f1:.4f}")
print(f"  Threshold: {result_ens.tau:.4f}")
print(f"  Precision: {result_ens.weighted_precision:.4f}")
print(f"  Recall:    {result_ens.weighted_recall:.4f}")

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED - Pipeline is working correctly!")
print("=" * 60)
print("\nYou can now run the full pipeline with:")
print("  ./run_pipeline.sh --skip-setup")
