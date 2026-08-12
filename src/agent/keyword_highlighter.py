import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class KeywordHighlighter:
    """
    AI Agent for Keyword Extraction and Visual Bounding Box Highlighting.
    Detects key concepts and draws color-coded overlays on document images.
    """
    def __init__(self):
        pass

    def extract_keywords(self, full_text, top_n=10):
        """
        Extract top important keywords and concepts using frequency and TF-IDF weights.
        """
        words = re.findall(r'\b[A-Za-z]{4,20}\b', full_text.lower())
        stopwords = {"this", "that", "with", "from", "have", "were", "been", "which", "their", "there", "about", "would", "these", "other"}
        filtered = [w for w in words if w not in stopwords]

        freq = {}
        for w in filtered:
            freq[w] = freq.get(w, 0) + 1

        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w[0] for w in sorted_words[:top_n]]

    def draw_highlights(self, pil_image, ocr_lines, target_keywords, highlight_color=(255, 235, 59, 120)):
        """
        Draw visual bounding box highlights directly on the PIL Image.
        Returns highlighted PIL Image.
        """
        img_rgb = pil_image.convert("RGBA")
        overlay = Image.new("RGBA", img_rgb.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        if isinstance(target_keywords, str):
            target_keywords = [target_keywords]

        keywords_lower = [k.lower() for k in target_keywords]

        highlight_count = 0
        for line in ocr_lines:
            text = line.get("text", "")
            bbox = line.get("bbox", [])
            
            if not bbox or len(bbox) < 4:
                continue

            x, y, w, h = bbox
            line_text_lower = text.lower()

            if any(k in line_text_lower for k in keywords_lower):
                draw.rectangle([x, y, x + w, y + h], fill=(255, 235, 59, 110), outline=(255, 152, 0, 230), width=2)
                highlight_count += 1

        combined = Image.alpha_composite(img_rgb, overlay)
        return combined.convert("RGB"), highlight_count
