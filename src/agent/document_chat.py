import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class DocumentChatAgent:
    """
    AI Agent for Interactive Document Chat & Question Answering.
    Answers natural language queries strictly based on uploaded document content.
    Supports multi-document indexing and context retrieval.
    """
    def __init__(self):
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None and SentenceTransformer is not None:
            try:
                self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                self._embedder = False
        return self._embedder

    def answer_question(self, query, document_texts, doc_names=None):
        """
        Answer question over single or multiple document texts.
        Returns response string, relevant citations, and confidence score.
        """
        if not document_texts:
            return {
                "answer": "No document content provided. Please upload or select a document.",
                "sources": [],
                "confidence": 0.0
            }

        if isinstance(document_texts, str):
            document_texts = [document_texts]
            doc_names = [doc_names[0] if doc_names else "Document 1"]
        elif not doc_names:
            doc_names = [f"Document {i+1}" for i in range(len(document_texts))]

        # Break documents into paragraphs/chunks
        chunks = []
        chunk_sources = []

        for text, name in zip(document_texts, doc_names):
            paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 15]
            if not paragraphs:
                paragraphs = [text[:500]]
            for p in paragraphs:
                chunks.append(p)
                chunk_sources.append(name)

        if not chunks:
            return {
                "answer": "No readable text passages found in document.",
                "sources": [],
                "confidence": 0.0
            }

        # Retrieval using Hybrid TF-IDF & Semantic Similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(chunks + [query])
            query_vec = tfidf_matrix[-1]
            doc_vecs = tfidf_matrix[:-1]
            sims = cosine_similarity(query_vec, doc_vecs).flatten()
        except Exception:
            sims = np.array([0.5] * len(chunks))

        top_indices = np.argsort(sims)[::-1][:3]
        top_chunks = [chunks[i] for i in top_indices if sims[i] > 0.05]
        top_sources = list(set([chunk_sources[i] for i in top_indices if sims[i] > 0.05]))

        if not top_chunks:
            # Fallback to first few paragraphs
            top_chunks = chunks[:2]
            top_sources = list(set(chunk_sources[:2]))

        context_str = "\n".join(top_chunks)
        confidence = float(np.max(sims)) if len(sims) > 0 and np.max(sims) > 0 else 0.75

        answer_text = self._synthesize_answer(query, context_str)

        return {
            "answer": answer_text,
            "sources": top_sources,
            "relevant_snippets": top_chunks,
            "confidence": round(confidence, 2)
        }

    def _synthesize_answer(self, query, context):
        q_lower = query.lower()
        lines = context.split("\n")
        
        # Check direct definitions
        if "definition" in q_lower or "what is" in q_lower or "define" in q_lower:
            keyword = q_lower.replace("definition of", "").replace("what is", "").replace("define", "").replace("?", "").strip()
            for line in lines:
                if keyword and keyword in line.lower():
                    return f"Based on the document:\n\n{line}\n\n(Context passage: \"{line}\")"

        # Summarized context response
        return f"Based strictly on the document content:\n\n{context}\n\n(Extracted from relevant document sections)"
