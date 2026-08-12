"""
train_trocr.py
--------------
Fine-tunes microsoft/trocr-base-printed on the OCR dataset word crops.

Pipeline:
  1. Load train_annot.parquet + test_annot.parquet (word-level, image IDs + labels)
  2. Build a PyTorch Dataset that crops each word from its parent image using bbox
  3. Fine-tune TrOCR (ViT encoder + RoBERTa decoder) for 3 epochs
  4. Evaluate CER on the test split
  5. Save fine-tuned model to models/trocr_finetuned/

Requirements:
    pip install transformers datasets torch torchvision Pillow jiwer

Usage:
    python src/train_trocr.py
"""

import os
import sys
import time
import random
import warnings
from pathlib import Path
from functools import lru_cache

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")

# ─── Configuration ─────────────────────────────────────────────────────────────
BASE_MODEL      = "microsoft/trocr-base-printed"
DATASET_DIR     = Path("dataset")
IMAGE_DIR       = DATASET_DIR / "train_val_images" / "train_images"
OUTPUT_DIR      = Path("models") / "trocr_finetuned"
TRAIN_FILE      = DATASET_DIR / "train_annot.parquet"
TEST_FILE       = DATASET_DIR / "test_annot.parquet"

BATCH_SIZE      = 8          # reduce to 4 if OOM on CPU
EPOCHS          = 3
LEARNING_RATE   = 5e-5
MAX_TARGET_LEN  = 32         # max tokens per word label
IMG_SIZE        = (384, 384) # TrOCR expected input
EVAL_SAMPLES    = 500        # number of test samples to compute CER on
MAX_TRAIN_STEPS = None       # set to int to cap steps (e.g. 1000) for quick test
RANDOM_SEED     = 42
# ──────────────────────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _build_image_map(image_dir: Path) -> dict:
    """
    Scan the image directory once and return a stem->full_path dict.
    Uses os.scandir() which is ~3-5x faster than Path.iterdir() on Windows.
    Cached globally so train + test datasets share the same map.
    """
    print(f"      Building image map from {image_dir} ...", end="", flush=True)
    img_map = {}
    with os.scandir(image_dir) as it:
        for entry in it:
            if entry.is_file():
                stem = os.path.splitext(entry.name)[0]
                img_map[stem] = Path(entry.path)
    print(f" done ({len(img_map):,} images indexed).")
    return img_map

_IMAGE_MAP_CACHE: dict | None = None  # module-level cache

def get_image_map(image_dir: Path) -> dict:
    """Return cached image map, building it only once."""
    global _IMAGE_MAP_CACHE
    if _IMAGE_MAP_CACHE is None:
        _IMAGE_MAP_CACHE = _build_image_map(image_dir)
    return _IMAGE_MAP_CACHE


# ─── Dataset ───────────────────────────────────────────────────────────────────
class OCRWordCropDataset(Dataset):
    """
    Crops individual words from their parent images using (x, y, w, h) bbox.
    Returns pixel_values (processed by ViTImageProcessor) and labels (tokenized text).
    """
    def __init__(self, df: pd.DataFrame, processor,
                 tokenizer, image_dir: Path, max_target_len: int = MAX_TARGET_LEN):
        self.df             = df.reset_index(drop=True)
        self.processor      = processor    # ViTImageProcessor
        self.tokenizer      = tokenizer    # RobertaTokenizer
        self.image_dir      = image_dir
        self.max_target_len = max_target_len
        # Use cached global image map (built once, shared across all dataset instances)
        self._img_map       = get_image_map(image_dir)

    def __len__(self):
        return len(self.df)

    def _load_crop(self, row) -> Image.Image:
        """Loads and crops a word patch from the full document image."""
        img_id   = row["image_id"]
        # Find matching file (stem may match id or part of filename)
        img_path = self._img_map.get(img_id)
        if img_path is None:
            # Fallback: glob by prefix
            candidates = list(self.image_dir.glob(f"{img_id}*"))
            img_path = candidates[0] if candidates else None

        if img_path is None or not img_path.exists():
            # Return blank white image if file not found
            return Image.new("RGB", (64, 32), color=(255, 255, 255))

        img = Image.open(img_path).convert("RGB")
        x, y, w, h = float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"])
        iw, ih = img.size
        # Clamp to image bounds
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(iw, int(x + w))
        y2 = min(ih, int(y + h))
        if x2 <= x1 or y2 <= y1:
            return img.resize((64, 32))
        crop = img.crop((x1, y1, x2, y2))
        return crop

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        crop  = self._load_crop(row)
        label = str(row["utf8_string"])

        # Encode image with ViTImageProcessor
        pixel_values = self.processor(
            images=crop.convert("RGB"), return_tensors="pt"
        ).pixel_values.squeeze(0)  # [C, H, W]

        # Encode label directly with tokenizer (transformers 5.x compatible)
        encoding = self.tokenizer(
            text=label,
            padding="max_length",
            max_length=self.max_target_len,
            truncation=True,
            return_tensors="pt",
        )
        labels = encoding.input_ids.squeeze(0)
        # Replace padding token id with -100 so loss ignores them
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


# ─── CER Evaluation — TrOCR ───────────────────────────────────────────────────
def compute_cer_trocr(model, image_processor, tokenizer, df_test: pd.DataFrame,
                      image_dir: Path, n_samples: int = EVAL_SAMPLES) -> float:
    """Compute Character Error Rate using the TrOCR model on test samples."""
    try:
        from jiwer import cer as jiwer_cer
    except ImportError:
        print("      [INFO] jiwer not installed; skipping CER. pip install jiwer")
        return -1.0

    model.eval()
    sample  = df_test.sample(min(n_samples, len(df_test)), random_state=RANDOM_SEED)
    dataset = OCRWordCropDataset(sample, image_processor, tokenizer, image_dir)

    preds_all, refs_all = [], []
    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            pv   = item["pixel_values"].unsqueeze(0).to(DEVICE)
            gen  = model.generate(pv, max_new_tokens=MAX_TARGET_LEN)
            pred = tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
            ref  = str(sample.iloc[idx]["utf8_string"]).strip()
            preds_all.append(pred if pred else " ")
            refs_all.append(ref if ref else " ")

    return jiwer_cer(refs_all, preds_all)


# ─── CER Evaluation — EasyOCR Fallback ────────────────────────────────────────
def compute_cer_easyocr(df_test: pd.DataFrame, image_dir: Path,
                        n_samples: int = EVAL_SAMPLES) -> float:
    """
    Fallback: compute CER using EasyOCR (pre-trained, no fine-tuning needed).
    Called automatically when TrOCR fails to load.
    """
    try:
        from jiwer import cer as jiwer_cer
    except ImportError:
        print("      [INFO] jiwer not installed; skipping CER. pip install jiwer")
        return -1.0

    import easyocr
    reader = easyocr.Reader(["en"], gpu=(DEVICE == "cuda"))
    img_map = get_image_map(image_dir)

    sample = df_test.sample(min(n_samples, len(df_test)), random_state=RANDOM_SEED)
    preds_all, refs_all = [], []

    for _, row in sample.iterrows():
        img_id   = row["image_id"]
        img_path = img_map.get(img_id)
        ref      = str(row["utf8_string"]).strip()

        if img_path is None or not img_path.exists():
            continue

        # Crop the word region from the full image
        img = Image.open(img_path).convert("RGB")
        x1 = max(0, int(row["x"]))
        y1 = max(0, int(row["y"]))
        x2 = min(img.width,  int(row["x"] + row["w"]))
        y2 = min(img.height, int(row["y"] + row["h"]))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img.crop((x1, y1, x2, y2))

        # EasyOCR inference on the cropped word
        results = reader.readtext(crop, detail=0)  # returns list of strings
        pred = " ".join(results).strip() if results else ""

        preds_all.append(pred if pred else " ")
        refs_all.append(ref  if ref  else " ")

    if not preds_all:
        return -1.0
    return jiwer_cer(refs_all, preds_all)


# ─── TrOCR Training Pipeline ──────────────────────────────────────────────────
def _train_trocr():
    """
    Full TrOCR fine-tuning pipeline.
    Raises on any error so the caller can fall back to EasyOCR.
    """
    # ── 1. Load TrOCR components ───────────────────────────────────────────────
    # Manual assembly for transformers 5.x compatibility.
    # TrOCRProcessor.from_pretrained() fails in 5.x with fast-tokenizer errors;
    # loading ViTImageProcessor + RobertaTokenizer(use_fast=False) separately works.
    print(f"\n[1/5] Loading TrOCR processor & model from {BASE_MODEL} ...")
    image_processor = ViTImageProcessor.from_pretrained(BASE_MODEL)
    tokenizer       = RobertaTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    model           = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)

    # Configure decoder for generation
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id           = tokenizer.pad_token_id
    model.config.vocab_size             = model.config.decoder.vocab_size
    model.to(DEVICE)
    print("      TrOCR model loaded successfully.")

    # ── 2. Build datasets ──────────────────────────────────────────────────────
    print(f"\n[2/5] Building datasets ...")
    df_train = pd.read_parquet(TRAIN_FILE)
    df_test  = pd.read_parquet(TEST_FILE)
    print(f"      Train words : {len(df_train):,}")
    print(f"      Test  words : {len(df_test):,}")

    if MAX_TRAIN_STEPS is not None:
        cap = MAX_TRAIN_STEPS * BATCH_SIZE
        df_train = df_train.sample(min(cap, len(df_train)), random_state=RANDOM_SEED)
        print(f"      [INFO] Capped train to {len(df_train):,} rows")

    train_dataset = OCRWordCropDataset(df_train, image_processor, tokenizer, IMAGE_DIR)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=0,
                               pin_memory=(DEVICE == "cuda"))
    print(f"      Train batches: {len(train_loader):,}")

    # ── 3. Optimizer & Scheduler ───────────────────────────────────────────────
    optimizer     = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps   = len(train_loader) * EPOCHS
    warmup_steps  = max(1, total_steps // 10)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── 4. Training loop ───────────────────────────────────────────────────────
    print(f"\n[3/5] Training TrOCR for {EPOCHS} epoch(s) ...")
    model.train()

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        t0 = time.time()
        for step, batch in enumerate(train_loader, 1):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels       = batch["labels"].to(DEVICE)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss    = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if step % 100 == 0 or step == len(train_loader):
                avg     = epoch_loss / step
                elapsed = time.time() - t0
                print(f"      Epoch {epoch}/{EPOCHS} | Step {step}/{len(train_loader)} "
                      f"| Loss: {avg:.4f} | Elapsed: {elapsed:.1f}s")

        print(f"\n  --> Epoch {epoch} done. Avg Loss: {epoch_loss/len(train_loader):.4f}\n")

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    print(f"[4/5] Evaluating TrOCR on {EVAL_SAMPLES} test samples ...")
    cer = compute_cer_trocr(model, image_processor, tokenizer, df_test, IMAGE_DIR)
    if cer >= 0:
        print(f"      TrOCR CER: {cer:.4f}  ({cer*100:.2f}%)")
    else:
        print("      CER skipped (jiwer not installed).")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    print(f"\n[5/5] Saving fine-tuned TrOCR to {OUTPUT_DIR} ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    image_processor.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"      Model saved -> {OUTPUT_DIR}")

    print("\n" + "=" * 60)
    print("  TROCR TRAINING COMPLETE!")
    if cer >= 0:
        print(f"  Final CER  : {cer*100:.2f}%")
    print(f"  Model saved: {OUTPUT_DIR}")
    print("=" * 60)


# ─── EasyOCR Fallback Pipeline ────────────────────────────────────────────────
def _eval_easyocr():
    """
    Fallback when TrOCR cannot be loaded.
    EasyOCR is pre-trained (no fine-tuning needed), so we just measure its
    baseline CER on the test split to report performance.
    """
    print("\n" + "=" * 60)
    print("  [FALLBACK] EASYOCR EVALUATION PIPELINE")
    print("  EasyOCR is pre-trained — measuring baseline CER on test set.")
    print("=" * 60)

    df_test = pd.read_parquet(TEST_FILE)
    print(f"\n  Test words loaded : {len(df_test):,}")
    print(f"  Evaluating on {EVAL_SAMPLES} samples ...")

    cer = compute_cer_easyocr(df_test, IMAGE_DIR, n_samples=EVAL_SAMPLES)
    if cer >= 0:
        print(f"\n  EasyOCR baseline CER : {cer:.4f}  ({cer*100:.2f}%)")
    else:
        print("\n  CER skipped (jiwer not installed).")

    print("\n" + "=" * 60)
    print("  EASYOCR EVALUATION COMPLETE")
    print("  NOTE: No model saved — EasyOCR uses its built-in pre-trained weights.")
    print("  Retry TrOCR training: pip install 'transformers<5' sentencepiece")
    print("=" * 60)


# ─── Entry Point ──────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("   OCR TRAINING PIPELINE")
    print(f"   Primary : TrOCR ({BASE_MODEL})")
    print(f"   Fallback: EasyOCR (pre-trained)")
    print(f"   Device  : {DEVICE.upper()}")
    print("=" * 60)

    # Check split files exist
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        print("\n[ERROR] Train/test split not found!")
        print("        Please run first:  python src/split_dataset.py")
        sys.exit(1)

    # ── Try TrOCR first ───────────────────────────────────────────────────────
    try:
        print("\n  Attempting TrOCR pipeline ...")
        _train_trocr()

    except Exception as trocr_err:
        # ── TrOCR failed → switch to EasyOCR ─────────────────────────────────
        print("\n" + "!" * 60)
        print(f"  [WARNING] TrOCR failed to load/train:")
        print(f"  {type(trocr_err).__name__}: {trocr_err}")
        print("  Switching to EasyOCR fallback ...")
        print("!" * 60)
        _eval_easyocr()


if __name__ == "__main__":
    train()
