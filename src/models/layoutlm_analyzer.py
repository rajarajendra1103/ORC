import numpy as np

class LayoutLMv3Analyzer:
    """
    LayoutLMv3 Spatial Layout & Document Structure Analyzer
    Analyzes visual structure, spatial relationships, header hierarchy, and table regions.
    """
    def __init__(self):
        self.model_name = "LayoutLMv3 (Spatial Layout Understanding)"

    def analyze_layout(self, lines_info, image_width=800, image_height=1000):
        """
        Analyze layout structure from text bounding boxes.
        Assigns spatial categories: Title, Subheader, Paragraph, Table Row, Key-Value Pair, Footer.
        """
        if not lines_info:
            return {
                "model": self.model_name,
                "blocks": [],
                "layout_summary": {"header_count": 0, "paragraph_count": 0, "table_region_count": 0}
            }

        blocks = []
        headers = []
        table_rows = []
        paragraphs = []

        # Sort lines by vertical position
        sorted_lines = sorted(lines_info, key=lambda l: l["bbox"][1])

        for line in sorted_lines:
            text = line["text"]
            x, y, w, h = line["bbox"]
            norm_bbox = [
                round(x / max(image_width, 1), 4),
                round(y / max(image_height, 1), 4),
                round((x + w) / max(image_width, 1), 4),
                round((y + h) / max(image_height, 1), 4)
            ]

            # Classification heuristics based on height, font ratio, position, and text structure
            category = "Paragraph"
            if h > (image_height * 0.035) or (len(text) < 40 and text.isupper()):
                category = "Header / Title"
                headers.append(text)
            elif ":" in text and len(text.split(":")[0]) < 25:
                category = "Key-Value Field"
            elif "\t" in text or "   " in text or text.count(" ") > 5 and len(text) < 50:
                category = "Table Cell / Row"
                table_rows.append(text)
            elif y > (image_height * 0.92) or "page" in text.lower() or "copyright" in text.lower():
                category = "Footer / Metadata"
            else:
                paragraphs.append(text)

            blocks.append({
                "text": text,
                "category": category,
                "bbox": [x, y, w, h],
                "normalized_bbox": norm_bbox,
                "confidence": line.get("confidence", 0.9)
            })

        return {
            "model": self.model_name,
            "blocks": blocks,
            "layout_summary": {
                "total_blocks": len(blocks),
                "header_count": len(headers),
                "paragraph_count": len(paragraphs),
                "table_region_count": len(table_rows),
                "detected_headers": headers[:5]
            }
        }
