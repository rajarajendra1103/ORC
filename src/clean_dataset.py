import os
import ast
import json
import pandas as pd
import numpy as np
from pathlib import Path

def clean_and_build_dataset(data_dir="dataset", output_dir="dataset"):
    print("=" * 60)
    print("      STARTING OCR DATASET CLEANING & PROCESSING PIPELINE")
    print("=" * 60)

    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Images Metadata
    img_parquet = data_path / "img.parquet"
    img_csv = data_path / "img.csv"
    
    if img_parquet.exists():
        print(f"[1/5] Loading image metadata from {img_parquet}...")
        df_img = pd.read_parquet(img_parquet)
    elif img_csv.exists():
        print(f"[1/5] Loading image metadata from {img_csv}...")
        df_img = pd.read_csv(img_csv)
    else:
        raise FileNotFoundError("Neither img.parquet nor img.csv found in dataset folder.")

    print(f"      Loaded {len(df_img):,} raw image records.")

    # Resolve physical image files
    img_dir = data_path / "train_val_images"
    existing_file_map = {}
    if img_dir.exists():
        for root, _, files in os.walk(img_dir):
            for f in files:
                rel_p = os.path.relpath(os.path.join(root, f), data_path).replace("\\", "/")
                existing_file_map[f] = rel_p

    def resolve_path(fn):
        bn = os.path.basename(fn)
        if bn in existing_file_map:
            return existing_file_map[bn]
        return fn.replace("\\", "/")

    df_img["resolved_file_path"] = df_img["file_name"].apply(resolve_path)
    df_img["file_exists"] = df_img["resolved_file_path"].apply(lambda p: (data_path.parent / p).exists() or (data_path / os.path.basename(p)).exists() or True)

    # Clean Image Metadata
    df_img_clean = df_img.drop_duplicates(subset=["id"]).copy()
    print(f"      Clean image records: {len(df_img_clean):,}")

    # Save Cleaned Image Datasets
    df_img_clean.to_parquet(out_path / "cleaned_img.parquet", index=False)
    df_img_clean.to_csv(out_path / "cleaned_img.csv", index=False)
    print(f"      [SAVED] {out_path / 'cleaned_img.parquet'} & {out_path / 'cleaned_img.csv'}")

    # 2. Load Annotations
    ann_parquet = data_path / "annot.parquet"
    ann_csv = data_path / "annot.csv"

    if ann_parquet.exists():
        print(f"\n[2/5] Loading annotations from {ann_parquet}...")
        df_ann = pd.read_parquet(ann_parquet)
    elif ann_csv.exists():
        print(f"\n[2/5] Loading annotations from {ann_csv}...")
        df_ann = pd.read_csv(ann_csv)
    else:
        raise FileNotFoundError("Neither annot.parquet nor annot.csv found in dataset folder.")

    raw_ann_count = len(df_ann)
    print(f"      Loaded {raw_ann_count:,} raw annotation records.")

    # 3. Clean Text & Bounding Boxes
    print("\n[3/5] Cleaning annotations (filtering illegible '.', nulls, degenerate bboxes)...")
    
    # Ensure utf8_string is string
    df_ann["utf8_string"] = df_ann["utf8_string"].fillna("").astype(str).str.strip()

    # Filter out illegible single dot (TextOCR standard for unreadable text) and empty strings
    legible_mask = (df_ann["utf8_string"] != ".") & (df_ann["utf8_string"] != "")
    df_ann_clean = df_ann[legible_mask].copy()

    filtered_illegible = raw_ann_count - len(df_ann_clean)
    print(f"      Filtered {filtered_illegible:,} illegible/empty annotations ({(filtered_illegible/raw_ann_count)*100:.2f}%).")

    # Parse bbox format [x, y, w, h]
    def parse_bbox(b):
        if isinstance(b, str):
            try:
                b = ast.literal_eval(b)
            except Exception:
                return [0.0, 0.0, 0.0, 0.0]
        if isinstance(b, (list, np.ndarray, tuple)) and len(b) >= 4:
            return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
        return [0.0, 0.0, 0.0, 0.0]

    df_ann_clean["bbox_parsed"] = df_ann_clean["bbox"].apply(parse_bbox)
    df_ann_clean["x"] = df_ann_clean["bbox_parsed"].apply(lambda b: b[0])
    df_ann_clean["y"] = df_ann_clean["bbox_parsed"].apply(lambda b: b[1])
    df_ann_clean["w"] = df_ann_clean["bbox_parsed"].apply(lambda b: b[2])
    df_ann_clean["h"] = df_ann_clean["bbox_parsed"].apply(lambda b: b[3])

    # Filter degenerate bounding boxes (w < 2 or h < 2 or area <= 0)
    valid_bbox_mask = (df_ann_clean["w"] >= 2.0) & (df_ann_clean["h"] >= 2.0) & (df_ann_clean["area"] > 0)
    df_ann_clean = df_ann_clean[valid_bbox_mask].copy()

    # Add normalized bounding box coordinates if image dimensions available
    img_dim_map = df_img_clean.set_index("id")[["width", "height"]].to_dict("index")
    
    def norm_bbox(row):
        img_info = img_dim_map.get(row["image_id"], {"width": 1.0, "height": 1.0})
        iw = max(float(img_info["width"]), 1.0)
        ih = max(float(img_info["height"]), 1.0)
        x, y, w, h = row["x"], row["y"], row["w"], row["h"]
        return [
            round(x / iw, 4),
            round(y / ih, 4),
            round((x + w) / iw, 4),
            round((y + h) / ih, 4)
        ]

    df_ann_clean["bbox_normalized"] = df_ann_clean.apply(norm_bbox, axis=1)

    print(f"      Total clean annotations remaining: {len(df_ann_clean):,}")

    # Save Cleaned Annotations
    cols_to_save = ["id", "image_id", "utf8_string", "x", "y", "w", "h", "area", "bbox_normalized"]
    df_ann_save = df_ann_clean[cols_to_save].copy()
    
    df_ann_save.to_parquet(out_path / "cleaned_annot.parquet", index=False)
    df_ann_save.to_csv(out_path / "cleaned_annot.csv", index=False)
    print(f"      [SAVED] {out_path / 'cleaned_annot.parquet'} & {out_path / 'cleaned_annot.csv'}")

    # 4. Spatial Line-Ordering & Document Corpus Reconstruction
    print("\n[4/5] Reconstructing document-level structured text corpus (Spatial Line Sorting)...")

    def sort_and_group_words(group):
        # Sort words primarily by vertical y position (binned into lines) and x position
        # Estimate line height median
        h_med = group["h"].median() if len(group) > 0 else 10.0
        line_threshold = max(h_med * 0.6, 5.0)

        # Sort by y first to cluster lines
        group_sorted = group.sort_values(by="y").copy()
        
        lines = []
        current_line = []
        current_y = None

        for _, row in group_sorted.iterrows():
            if current_y is None or abs(row["y"] - current_y) <= line_threshold:
                current_line.append(row)
                if current_y is None:
                    current_y = row["y"]
                else:
                    # Update moving average y for line
                    current_y = current_y * 0.7 + row["y"] * 0.3
            else:
                # Sort current line left-to-right by x
                current_line.sort(key=lambda r: r["x"])
                lines.append(" ".join([r["utf8_string"] for r in current_line]))
                current_line = [row]
                current_y = row["y"]

        if current_line:
            current_line.sort(key=lambda r: r["x"])
            lines.append(" ".join([r["utf8_string"] for r in current_line]))

        full_doc_text = "\n".join(lines)
        word_count = len(group)
        return pd.Series({
            "full_document_text": full_doc_text,
            "word_count": word_count,
            "line_count": len(lines)
        })

    doc_corpus = df_ann_clean.groupby("image_id").apply(sort_and_group_words, include_groups=False).reset_index()
    
    # Merge with image metadata
    doc_corpus = doc_corpus.merge(df_img_clean[["id", "file_name", "resolved_file_path", "width", "height", "set"]], 
                                  left_on="image_id", right_on="id", how="left").drop(columns=["id"])

    print(f"      Total processed documents in corpus: {len(doc_corpus):,}")
    print(f"      Avg word count per document: {doc_corpus['word_count'].mean():.1f} words")

    # Save Document Corpus
    doc_corpus.to_parquet(out_path / "document_ocr_corpus.parquet", index=False)
    doc_corpus.to_csv(out_path / "document_ocr_corpus.csv", index=False)
    
    jsonl_path = out_path / "document_ocr_corpus.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in doc_corpus.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"      [SAVED] {out_path / 'document_ocr_corpus.parquet'}")
    print(f"      [SAVED] {out_path / 'document_ocr_corpus.csv'}")
    print(f"      [SAVED] {out_path / 'document_ocr_corpus.jsonl'}")

    # 5. Summary Statistics Report
    print("\n[5/5] CLEANING & PROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
    print("-" * 60)
    print(f"  - Raw Annotations:           {raw_ann_count:,}")
    print(f"  - Cleaned Annotations:       {len(df_ann_save):,}")
    print(f"  - Total Cleaned Images:      {len(df_img_clean):,}")
    print(f"  - Clean Document Corpus:     {len(doc_corpus):,} documents")
    print("=" * 60)

if __name__ == "__main__":
    clean_and_build_dataset()
