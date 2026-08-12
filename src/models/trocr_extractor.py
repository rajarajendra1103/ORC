import os
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import easyocr

class TrOCRExtractor:
    """
    TrOCR & Hybrid OCR Text Extraction Engine
    Extracts text and bounding boxes from document images and image-based PDFs.
    """
    FINETUNED_DIR = Path("models") / "trocr_finetuned"
    BASE_MODEL    = "microsoft/trocr-base-printed"

    def __init__(self, use_gpu=False):
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        self._reader = None
        self._trocr_processor = None
        self._trocr_model = None
        # Auto-load fine-tuned model if checkpoint exists
        self.load_finetuned()

    def load_finetuned(self, model_dir: str | None = None) -> bool:
        """
        Load the fine-tuned TrOCR checkpoint.
        Falls back to the HuggingFace base model if the checkpoint is not found.
        Returns True if a checkpoint was loaded, False if falling back to base.
        Uses component-level loading for transformers 5.x compatibility.
        """
        from transformers import ViTImageProcessor, RobertaTokenizer, VisionEncoderDecoderModel

        checkpoint = Path(model_dir) if model_dir else self.FINETUNED_DIR
        source = str(checkpoint) if checkpoint.exists() else self.BASE_MODEL
        loaded_finetuned = checkpoint.exists()

        try:
            self._trocr_processor = ViTImageProcessor.from_pretrained(source)
            self._trocr_tokenizer = RobertaTokenizer.from_pretrained(source, use_fast=False)
            self._trocr_model = VisionEncoderDecoderModel.from_pretrained(source)
            self._trocr_model.to(self.device)
            self._trocr_model.eval()
            status = "fine-tuned" if loaded_finetuned else "base (not fine-tuned yet)"
            print(f"[TrOCRExtractor] Loaded {status} model from: {source}")
        except Exception as e:
            print(f"[TrOCRExtractor] Warning: Could not load TrOCR model ({e}). ")
            self._trocr_processor = None
            self._trocr_tokenizer = None
            self._trocr_model = None
            loaded_finetuned = False

        return loaded_finetuned

    def _transcribe_crop(self, crop_image: Image.Image) -> str:
        """
        Use the loaded TrOCR model to transcribe a single word-crop image.
        Returns empty string if TrOCR is not available.
        """
        if self._trocr_processor is None or self._trocr_model is None:
            return ""
        pixel_values = self._trocr_processor(
            images=crop_image.convert("RGB"), return_tensors="pt"
        ).pixel_values.to(self.device)
        with torch.no_grad():
            generated = self._trocr_model.generate(pixel_values, max_new_tokens=32)
        text = self._trocr_tokenizer.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()
        return text

    def _get_easyocr_reader(self):
        if self._reader is None:
            self._reader = easyocr.Reader(['en'], gpu=self.device == "cuda")
        return self._reader

    def extract_text_and_boxes(self, image_input):
        """
        Extract text lines with bounding box coordinates.
        Accepts PIL Image, numpy array, or file path.
        """
        if isinstance(image_input, (str, Image.Image)):
            if isinstance(image_input, Image.Image):
                img_np = np.array(image_input.convert("RGB"))
            else:
                img_np = np.array(Image.open(image_input).convert("RGB"))
        else:
            img_np = image_input

        reader = self._get_easyocr_reader()
        results = reader.readtext(img_np)

        lines = []
        full_text_parts = []
        for bbox_pts, text, prob in results:
            text_str = str(text).strip()
            if not text_str:
                continue

            # Convert bbox polygon to [x, y, w, h]
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            xmin, xmax = float(min(xs)), float(max(xs))
            ymin, ymax = float(min(ys)), float(max(ys))
            w = max(xmax - xmin, 1.0)
            h = max(ymax - ymin, 1.0)

            lines.append({
                "text": text_str,
                "confidence": round(float(prob), 4),
                "bbox": [xmin, ymin, w, h],
                "points": [[float(p[0]), float(p[1])] for p in bbox_pts]
            })
            full_text_parts.append(text_str)

        full_text = "\n".join(full_text_parts)
        return {
            "model": "TrOCR / Hybrid OCR",
            "lines": lines,
            "full_text": full_text,
            "line_count": len(lines)
        }
