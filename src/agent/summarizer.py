import re

class DocumentSummarizer:
    """
    AI Agent for Automatic Document Summarization
    Generates executive summaries, bullet points, and topic-wise summaries.
    """
    def __init__(self):
        pass

    def generate_summary(self, full_text, mode="all"):
        """
        Generate document summary in requested format: 'executive', 'bullets', 'topics', or 'all'.
        """
        paragraphs = [p.strip() for p in full_text.split("\n") if len(p.strip()) > 20]
        if not paragraphs:
            return {
                "executive_summary": "Document contains minimal text content.",
                "key_points": ["Insufficient text for bullet points."],
                "topic_summaries": {}
            }

        # Executive Summary: Extract top key sentences
        exec_summary = " ".join(paragraphs[:3]) if len(paragraphs) >= 3 else full_text

        # Key Bullet Points
        bullets = []
        for p in paragraphs:
            if len(p) > 35 and ("is" in p or "are" in p or "includes" in p or "total" in p or "provides" in p or "important" in p):
                bullets.append(p[:150] + ("..." if len(p) > 150 else ""))
            if len(bullets) >= 6:
                break
        if not bullets:
            bullets = [p[:150] for p in paragraphs[:5]]

        # Topic-wise Summaries (detecting headers or chapters)
        topics = {}
        curr_topic = "General Overview"
        topics[curr_topic] = []

        for p in paragraphs:
            if len(p) < 50 and (p.isupper() or p.startswith("Chapter") or p.startswith("Section") or p.endswith(":")):
                curr_topic = p.rstrip(":")
                topics[curr_topic] = []
            else:
                topics[curr_topic].append(p)

        topic_summaries = {t: " ".join(content[:2]) if content else "No detailed text under topic." for t, content in topics.items()}

        return {
            "executive_summary": exec_summary,
            "key_points": bullets,
            "topic_summaries": topic_summaries
        }
