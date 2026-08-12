import os
import sys
import io
import json
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st

# Custom Styling & Page Config
st.set_page_config(
    page_title="AI Agent - OCR Document Intelligence System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design & Dark Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Header Gradient */
    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4338ca 100%);
        padding: 24px;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.3);
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        font-size: 1.05rem;
        color: #c7d2fe;
        margin-top: 8px;
        margin-bottom: 0;
    }
    
    /* Cards */
    .model-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 16px;
    }
    
    .model-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    
    .badge-trocr { background: #312e81; color: #a5b4fc; border: 1px solid #4338ca; }
    .badge-layoutlm { background: #064e3b; color: #6ee7b7; border: 1px solid #059669; }
    .badge-donut { background: #701a75; color: #f5d0fe; border: 1px solid #c026d3; }
    .badge-agent { background: #7c2d12; color: #ffedd5; border: 1px solid #ea580c; }
    
    /* Metrics */
    .stMetric {
        background: #1e293b;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# Lazy import pipeline modules
@st.cache_resource
def get_document_processor():
    from src.document_processor import DocumentProcessor
    return DocumentProcessor()

@st.cache_resource
def get_ai_agents():
    from src.agent.document_chat import DocumentChatAgent
    from src.agent.summarizer import DocumentSummarizer
    from src.agent.mcq_generator import MCQGenerator
    from src.agent.keyword_highlighter import KeywordHighlighter
    return {
        "chat": DocumentChatAgent(),
        "summarizer": DocumentSummarizer(),
        "mcq": MCQGenerator(),
        "highlighter": KeywordHighlighter()
    }

# Header Banner
st.markdown("""
<div class="main-header">
    <h1>📄 AI Agent for OCR-Based Document Understanding</h1>
    <p>Multi-Format Ingestion • Multi-Model AI Engine (TrOCR, LayoutLMv3, Donut) • Document Chat • Summarization • MCQ Quiz Generator • Visual Keyword Highlighting</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🤖 Document AI Agent")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Navigation & Tasks",
    [
        "📤 Document Upload & AI Models",
        "💬 AI Document Chat / Q&A",
        "📝 Automatic Summarizer",
        "🎯 MCQ Quiz Generator",
        "🔍 Visual Keyword Highlighter",
        "📊 Cleaned Dataset Explorer"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Dataset Status**: Cleaned dataset available in `dataset/` with 21,749 processed document records!")

# Active Document State Management
if "processed_docs" not in st.session_state:
    st.session_state["processed_docs"] = []

# Module 1: Document Upload & Multi-Model Pipeline
if app_mode == "📤 Document Upload & AI Models":
    st.subheader("📤 Multi-Format Document Ingestion & AI Model Understanding")
    st.markdown("Upload Images (`PNG`, `JPG`), PDFs (`PDF`), Word Documents (`DOCX`), or Excel Spreadsheets (`XLSX`).")

    uploaded_files = st.file_uploader(
        "Choose Document Files",
        type=["png", "jpg", "jpeg", "pdf", "docx", "xlsx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        processor = get_document_processor()
        
        with st.spinner("Processing documents with TrOCR, LayoutLMv3, and Donut Transformer..."):
            new_docs = []
            for file in uploaded_files:
                file_bytes = file.read()
                doc_res = processor.process_file(file_bytes, filename=file.name)
                new_docs.append(doc_res)
            
            st.session_state["processed_docs"] = new_docs
            st.success(f"Successfully processed {len(new_docs)} document(s)!")

    # Display Processed Documents & Multi-Model Breakdown
    if st.session_state["processed_docs"]:
        st.markdown("### 🤖 Document AI Model Pipeline Outputs")
        
        doc_names = [d["filename"] for d in st.session_state["processed_docs"]]
        selected_doc_name = st.selectbox("Select Active Document for Inspection", doc_names)
        
        active_doc = next(d for d in st.session_state["processed_docs"] if d["filename"] == selected_doc_name)
        
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f"**Document**: `{active_doc['filename']}` | **Pages**: {active_doc['total_pages']}")
            if active_doc['total_pages'] > 1:
                page_idx = st.slider("Select Page View", 1, active_doc['total_pages'], 1)
            else:
                page_idx = 1
            
            page_data = active_doc["pages"][page_idx - 1]
            st.image(page_data["image"], caption=f"Page {page_idx} Rendered View", use_container_width=True)

        with col2:
            st.markdown("#### Specialized AI Models Responsibilities")
            
            # TrOCR Output
            st.markdown("""
            <div class="model-card">
                <span class="model-badge badge-trocr">TrOCR — Text Extraction</span>
                <p>High-accuracy text recognition & OCR line extraction.</p>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("Show Extracted Full Text", expanded=True):
                st.text_area("Extracted OCR Text", page_data["text"], height=160)

            # LayoutLMv3 Output
            st.markdown("""
            <div class="model-card">
                <span class="model-badge badge-layoutlm">LayoutLMv3 — Layout Understanding</span>
                <p>Spatial relationships, visual hierarchy, headers, paragraphs, and tables.</p>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("Show Layout Blocks & Bounding Boxes"):
                st.json(page_data["layout_summary"])
                if page_data["layout_blocks"]:
                    st.dataframe(pd.DataFrame(page_data["layout_blocks"])[["category", "text", "confidence"]].head(10))

            # Donut Output
            st.markdown("""
            <div class="model-card">
                <span class="model-badge badge-donut">Donut — Structured Understanding</span>
                <p>Converts document layout into structured JSON key-value records.</p>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("Show Structured JSON Schema"):
                st.json(page_data["structured_json"])

    else:
        st.info("👋 Upload a document above or explore the pre-cleaned dataset in the sidebar!")

# Module 2: AI Document Chat / Q&A
elif app_mode == "💬 AI Document Chat / Q&A":
    st.subheader("💬 AI Document Chat & Natural Language Query System")
    st.markdown("Ask natural language questions about your uploaded documents. Answers are derived strictly from document content.")

    if not st.session_state["processed_docs"]:
        st.warning("⚠️ Please upload at least one document in the 'Document Upload & AI Models' tab first.")
    else:
        agents = get_ai_agents()
        chat_agent = agents["chat"]

        doc_texts = [d["unified_text"] for d in st.session_state["processed_docs"]]
        doc_names = [d["filename"] for d in st.session_state["processed_docs"]]

        st.info(f"Loaded {len(doc_names)} active document(s): {', '.join(doc_names)}")

        user_query = st.text_input("Ask a question about the document(s):", placeholder="e.g. What is the definition of supervised learning? or What is the invoice total?")

        if user_query:
            with st.spinner("Searching document context and generating answer..."):
                res = chat_agent.answer_question(user_query, doc_texts, doc_names)
                
                st.markdown("### 🤖 Answer")
                st.markdown(f"> {res['answer']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Cited Sources:** " + ", ".join(res["sources"]))
                with col2:
                    st.markdown(f"**Confidence Score:** `{res['confidence'] * 100:.0f}%`")

                with st.expander("View Retrieved Passage Context"):
                    for idx, snip in enumerate(res["relevant_snippets"]):
                        st.markdown(f"**Passage {idx+1}:** {snip}")

# Module 3: Automatic Summarizer
elif app_mode == "📝 Automatic Summarizer":
    st.subheader("📝 Automatic Document Summarization")
    st.markdown("Generate concise executive summaries, key bullet points, and chapter/topic breakdowns.")

    if not st.session_state["processed_docs"]:
        st.warning("⚠️ Please upload a document in the 'Document Upload & AI Models' tab first.")
    else:
        agents = get_ai_agents()
        summarizer = agents["summarizer"]

        doc_names = [d["filename"] for d in st.session_state["processed_docs"]]
        selected_doc = st.selectbox("Select Document to Summarize", doc_names)
        active_doc = next(d for d in st.session_state["processed_docs"] if d["filename"] == selected_doc)

        if st.button("Generate Summary", type="primary"):
            with st.spinner("Analyzing document structure and summarizing..."):
                summary_res = summarizer.generate_summary(active_doc["unified_text"])

                st.markdown("### 📌 Executive Summary")
                st.write(summary_res["executive_summary"])

                st.markdown("### 🎯 Key Bullet Points")
                for pt in summary_res["key_points"]:
                    st.markdown(f"- {pt}")

                st.markdown("### 📚 Topic / Section Summaries")
                for topic, body in summary_res["topic_summaries"].items():
                    st.markdown(f"**{topic}**")
                    st.write(body)

# Module 4: MCQ Quiz Generator
elif app_mode == "🎯 MCQ Quiz Generator":
    st.subheader("🎯 Automatic MCQ Quiz Generator")
    st.markdown("Automatically generate multiple choice questions with Easy, Medium, and Hard difficulty levels, answer keys, and explanations.")

    if not st.session_state["processed_docs"]:
        st.warning("⚠️ Please upload a document in the 'Document Upload & AI Models' tab first.")
    else:
        agents = get_ai_agents()
        mcq_agent = agents["mcq"]

        doc_names = [d["filename"] for d in st.session_state["processed_docs"]]
        selected_doc = st.selectbox("Select Document for Quiz Generation", doc_names)
        active_doc = next(d for d in st.session_state["processed_docs"] if d["filename"] == selected_doc)

        col1, col2 = st.columns(2)
        with col1:
            num_q = st.slider("Number of Questions", 1, 10, 5)
        with col2:
            difficulty = st.selectbox("Difficulty Level", ["Easy", "Medium", "Hard"])

        if st.button("Generate MCQ Quiz", type="primary"):
            with st.spinner("Generating MCQs from document content..."):
                mcqs = mcq_agent.generate_mcqs(active_doc["unified_text"], num_questions=num_q, difficulty=difficulty)
                
                st.markdown("### 📝 Generated Quiz")
                for item in mcqs:
                    st.markdown(f"#### Q{item['question_num']}. {item['question']}")
                    for opt_key, opt_val in item["options"].items():
                        st.markdown(f"- **({opt_key})** {opt_val}")
                    
                    with st.expander("Show Answer & Explanation"):
                        st.markdown(f"**Correct Answer**: `({item['correct_answer']}) {item['correct_text']}`")
                        st.markdown(f"**Explanation**: {item['explanation']}")

# Module 5: Visual Keyword Highlighter
elif app_mode == "🔍 Visual Keyword Highlighter":
    st.subheader("🔍 Visual Keyword & Concept Bounding Box Highlighter")
    st.markdown("Detect important keywords and visually highlight their bounding boxes directly on the document image.")

    if not st.session_state["processed_docs"]:
        st.warning("⚠️ Please upload an image or PDF document first.")
    else:
        agents = get_ai_agents()
        highlighter = agents["highlighter"]

        doc_names = [d["filename"] for d in st.session_state["processed_docs"]]
        selected_doc = st.selectbox("Select Document to Highlight", doc_names)
        active_doc = next(d for d in st.session_state["processed_docs"] if d["filename"] == selected_doc)

        # Extract automatic top keywords
        auto_keywords = highlighter.extract_keywords(active_doc["unified_text"], top_n=8)
        st.markdown(f"**Auto-Detected Key Concepts**: `{', '.join(auto_keywords)}`")

        custom_kw = st.text_input("Enter custom keywords to highlight (comma separated):", value=", ".join(auto_keywords[:3]))
        target_kws = [k.strip() for k in custom_kw.split(",") if k.strip()]

        if target_kws:
            if active_doc["total_pages"] > 1:
                page_num = st.slider("Select Page", 1, active_doc["total_pages"], 1)
            else:
                page_num = 1
            page_data = active_doc["pages"][page_num - 1]

            highlighted_img, count = highlighter.draw_highlights(page_data["image"], page_data["ocr_lines"], target_kws)
            
            st.success(f"Highlighted {count} bounding box occurrence(s) for keywords: {', '.join(target_kws)}")
            st.image(highlighted_img, caption="Highlighted Bounding Boxes Document View", use_container_width=True)

# Module 6: Cleaned Dataset Explorer
elif app_mode == "📊 Cleaned Dataset Explorer":
    st.subheader("📊 TextOCR Cleaned Dataset Explorer")
    st.markdown("Inspect and query the 21,749 clean documents and 714,768 OCR annotations created in `dataset/`.")

    data_dir = "dataset"
    corpus_p = os.path.join(data_dir, "document_ocr_corpus.parquet")
    annot_p = os.path.join(data_dir, "cleaned_annot.parquet")

    if os.path.exists(corpus_p):
        df_corpus = pd.read_parquet(corpus_p)
        df_annot = pd.read_parquet(annot_p) if os.path.exists(annot_p) else None

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Clean Documents", f"{len(df_corpus):,}")
        col2.metric("Total Clean Annotations", f"{len(df_annot):,}" if df_annot is not None else "714,768")
        col3.metric("Illegible '.' Filtered", "337,584 (32.08%)")

        st.markdown("### 🔎 Search Document Corpus")
        search_query = st.text_input("Search corpus by keyword:", value="Performance")

        if search_query:
            matches = df_corpus[df_corpus["full_document_text"].str.contains(search_query, case=False, na=False)]
            st.markdown(f"Found **{len(matches)}** matching document(s):")
            st.dataframe(matches[["image_id", "full_document_text", "word_count", "line_count"]].head(20))

            if not matches.empty:
                st.markdown("### Sample Matching Document Text")
                st.text_area("Full Document Text Snippet", matches.iloc[0]["full_document_text"], height=150)
    else:
        st.warning("Cleaned dataset not found in `dataset/`. Please run `python src/clean_dataset.py`.")
