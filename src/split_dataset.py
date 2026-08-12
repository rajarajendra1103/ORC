"""
split_dataset.py
-----------------
Splits the cleaned OCR dataset into 80% train / 20% test at the IMAGE level.
This ensures all word annotations from the same image stay in the same split
(prevents data leakage).

Outputs saved to dataset/:
  - train_corpus.parquet   / test_corpus.parquet   (document-level)
  - train_annot.parquet    / test_annot.parquet     (word-level, for TrOCR)

Usage:
    python src/split_dataset.py
"""

import os
import random
import pandas as pd
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────
DATASET_DIR   = Path("dataset")
CORPUS_FILE   = DATASET_DIR / "document_ocr_corpus.parquet"
ANNOT_FILE    = DATASET_DIR / "cleaned_annot.parquet"
TRAIN_RATIO   = 0.80
RANDOM_SEED   = 42
# ──────────────────────────────────────────────────────────────────────────────


def split_dataset():
    print("=" * 60)
    print("     OCR DATASET  ->  80 / 20  TRAIN / TEST  SPLIT")
    print("=" * 60)

    # 1. Load corpus
    print(f"\n[1/4] Loading document corpus from {CORPUS_FILE} ...")
    df_corpus = pd.read_parquet(CORPUS_FILE)
    print(f"      Loaded {len(df_corpus):,} documents.")

    # 2. Split image IDs
    image_ids = df_corpus["image_id"].unique().tolist()
    random.seed(RANDOM_SEED)
    random.shuffle(image_ids)

    split_idx    = int(len(image_ids) * TRAIN_RATIO)
    train_ids    = set(image_ids[:split_idx])
    test_ids     = set(image_ids[split_idx:])

    print(f"\n[2/4] Splitting {len(image_ids):,} images ...")
    print(f"      Train images : {len(train_ids):,}  ({TRAIN_RATIO*100:.0f}%)")
    print(f"      Test  images : {len(test_ids):,}  ({(1-TRAIN_RATIO)*100:.0f}%)")

    # 3. Split corpus
    train_corpus = df_corpus[df_corpus["image_id"].isin(train_ids)].copy()
    test_corpus  = df_corpus[df_corpus["image_id"].isin(test_ids)].copy()

    train_corpus_path = DATASET_DIR / "train_corpus.parquet"
    test_corpus_path  = DATASET_DIR / "test_corpus.parquet"
    train_corpus.to_parquet(train_corpus_path, index=False)
    test_corpus.to_parquet(test_corpus_path,  index=False)
    print(f"\n[3/4] Corpus splits saved:")
    print(f"      OK  {train_corpus_path}  ({len(train_corpus):,} docs)")
    print(f"      OK  {test_corpus_path}   ({len(test_corpus):,} docs)")

    # 4. Split word-level annotations
    print(f"\n[4/4] Loading & splitting word annotations from {ANNOT_FILE} ...")
    df_annot = pd.read_parquet(ANNOT_FILE)
    print(f"      Loaded {len(df_annot):,} word annotations.")

    train_annot = df_annot[df_annot["image_id"].isin(train_ids)].copy()
    test_annot  = df_annot[df_annot["image_id"].isin(test_ids)].copy()

    train_annot_path = DATASET_DIR / "train_annot.parquet"
    test_annot_path  = DATASET_DIR / "test_annot.parquet"
    train_annot.to_parquet(train_annot_path, index=False)
    test_annot.to_parquet(test_annot_path,  index=False)
    print(f"      OK  {train_annot_path}  ({len(train_annot):,} words)")
    print(f"      OK  {test_annot_path}   ({len(test_annot):,} words)")

    # Summary
    print("\n" + "=" * 60)
    print("  SPLIT COMPLETE - SUMMARY")
    print("-" * 60)
    print(f"  Total documents  : {len(df_corpus):,}")
    print(f"  Train documents  : {len(train_corpus):,}  ({len(train_corpus)/len(df_corpus)*100:.1f}%)")
    print(f"  Test  documents  : {len(test_corpus):,}   ({len(test_corpus)/len(df_corpus)*100:.1f}%)")
    print(f"  Total words      : {len(df_annot):,}")
    print(f"  Train words      : {len(train_annot):,}  ({len(train_annot)/len(df_annot)*100:.1f}%)")
    print(f"  Test  words      : {len(test_annot):,}   ({len(test_annot)/len(df_annot)*100:.1f}%)")
    print("=" * 60)
    print("  Next step -> python src/train_trocr.py")
    print("=" * 60)


if __name__ == "__main__":
    split_dataset()
