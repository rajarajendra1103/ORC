import torch
from PIL import Image
import numpy as np
import easyocr

class TrOCRExtractor:
    """
    TrOCR & Hybrid OCR Text Extraction Engine
    Extracts text and bounding boxes from document images and image-based PDFs.
    """
    def __init__(self, use_gpu=False):
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        self._reader = None
        self._trocr_processor = None
        self._trocr_model = None

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
