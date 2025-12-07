"""
Data ingestion module for CAFA 6.

Handles loading sequences (FASTA), terms (TSV), and preparing labels.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_sequences(fasta_path: str, parse_uniprot: bool = True) -> Dict[str, str]:
    """
    Load protein sequences from a FASTA file.

    Args:
        fasta_path: Path to the FASTA file.
        parse_uniprot: If True, extract UniProt ID from headers like 'sp|P12345|NAME'.

    Returns:
        Dictionary mapping protein IDs to sequences.
    """
    sequences = {}

    try:
        from Bio import SeqIO

        for record in SeqIO.parse(fasta_path, "fasta"):
            if parse_uniprot and "|" in record.id:
                pid = record.id.split("|")[1]
            else:
                pid = record.id
            sequences[pid] = str(record.seq)
    except ImportError:
        # Fallback: manual FASTA parsing
        current_id = None
        current_seq = []

        with open(fasta_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id:
                        sequences[current_id] = "".join(current_seq)
                    header = line[1:]
                    if parse_uniprot and "|" in header:
                        current_id = header.split("|")[1]
                    else:
                        current_id = header.split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)

            if current_id:
                sequences[current_id] = "".join(current_seq)

    return sequences


def load_terms(tsv_path: str) -> pd.DataFrame:
    """
    Load GO term annotations from TSV file.

    Expected columns: EntryID, term, aspect (optional)

    Args:
        tsv_path: Path to the TSV file.

    Returns:
        DataFrame with term annotations.
    """
    return pd.read_csv(tsv_path, sep="\t")


def get_top_go_terms(
    terms_df: pd.DataFrame, n_terms: int = 1500, term_col: str = "term"
) -> List[str]:
    """
    Get the N most frequent GO terms.

    Args:
        terms_df: DataFrame with term annotations.
        n_terms: Number of top terms to select.
        term_col: Column name containing GO terms.

    Returns:
        List of top GO term IDs.
    """
    term_counts = terms_df[term_col].value_counts()
    return term_counts.head(n_terms).index.tolist()


def prepare_labels(
    protein_ids: List[str],
    terms_df: pd.DataFrame,
    top_go_terms: List[str],
    entry_col: str = "EntryID",
    term_col: str = "term",
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Create binary label matrix for proteins.

    Args:
        protein_ids: List of protein IDs (determines row order).
        terms_df: DataFrame with term annotations.
        top_go_terms: List of GO terms to include (determines column order).
        entry_col: Column name for protein IDs.
        term_col: Column name for GO terms.

    Returns:
        Tuple of (label_matrix, go_term_to_index mapping).
    """
    go2idx = {go: i for i, go in enumerate(top_go_terms)}
    pid2idx = {pid: i for i, pid in enumerate(protein_ids)}

    labels = np.zeros((len(protein_ids), len(top_go_terms)), dtype=np.float32)

    # Group terms by protein
    labels_dict = terms_df.groupby(entry_col)[term_col].apply(list).to_dict()

    for pid, terms in tqdm(labels_dict.items(), desc="Building label matrix"):
        if pid not in pid2idx:
            continue
        row_idx = pid2idx[pid]
        for term in terms:
            if term in go2idx:
                labels[row_idx, go2idx[term]] = 1.0

    return labels, go2idx


def load_ia_weights(ia_file: str, go_terms: List[str]) -> np.ndarray:
    """
    Load Information Accretion (IA) weights for GO terms.

    Args:
        ia_file: Path to JSON file with IA weights.
        go_terms: List of GO terms (determines order).

    Returns:
        Array of IA weights, same order as go_terms.
    """
    import json

    with open(ia_file, "r") as f:
        ia_dict = json.load(f)

    weights = np.ones(len(go_terms), dtype=np.float32)
    for i, term in enumerate(go_terms):
        if term in ia_dict:
            weights[i] = ia_dict[term]

    return weights
