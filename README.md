# AI Agent for OCR-Based Document Understanding and Intelligent Query System

An intelligent document understanding and interactive query system powered by multi-model vision-language architectures (**TrOCR**, **LayoutLMv3**, **Donut Transformer**) and an autonomous AI Document Agent.

---

## 🌟 Key Features

1. **Modern HTML5 + Vanilla CSS + JavaScript Web Application**:
   - Ultra-sleek dark aesthetics with **high-contrast, bright text** (`#FFFFFF`, `#F8FAFC`) on deep dark backgrounds.
   - Glassmorphism containers, smooth CSS transitions, glowing badges, and micro-interactions.
   - Powered by a high-performance **FastAPI** Python REST server.

2. **Multi-Format Document Upload**:
   - Images (`PNG`, `JPG`, `JPEG`, `BMP`, `TIFF`)
   - PDFs (`PDF` with native text layer extraction and image rendering)
   - Word Documents (`DOC`, `DOCX`)
   - Excel Spreadsheets (`XLS`, `XLSX`)

3. **Specialized Document AI Models**:
   - **TrOCR — Text Extraction**: High-accuracy optical character recognition & text extraction from scanned document images and image-based PDFs.
   - **LayoutLMv3 — Layout Understanding**: Analyzes visual hierarchy, spatial bounding box relationships, headers, subheaders, paragraphs, footers, and table regions.
   - **Donut Transformer — Structured Understanding**: Converts unstructured document images into structured JSON key-value records.

4. **AI Document Agent Capabilities**:
   - **💬 AI Document Chat / Q&A**: Interactive natural language Q&A over single or multiple uploaded documents, grounded strictly in document content.
   - **📝 Automatic Summarization**: Executive summaries, important bullet points, and chapter/topic breakdowns.
   - **🎯 MCQ Quiz Generator**: Automated Quiz/MCQ generator across `Easy`, `Medium`, and `Hard` difficulty levels with answer keys and explanations.
   - **🔍 Visual Keyword Highlighter**: Auto-detects key concepts and draws color-coded bounding box highlights directly on document pages.

5. **Cleaned TextOCR Benchmark Dataset**:
   - Filtered out 337,584 single-dot (`.`) illegible annotations and degenerate bounding boxes ($w < 2\text{px}, h < 2\text{px}$).
   - Spatial line-ordering reconstructs 21,749 clean document texts.
   - Generates `cleaned_annot.parquet`, `cleaned_img.parquet`, and `document_ocr_corpus.parquet`.

---

## 🚀 Quick Start & Installation

### 1. Install Dependencies
```bash
pip install torch transformers sentence-transformers easyocr pillow python-docx openpyxl pypdf pymupdf pandas pyarrow fastapi uvicorn scikit-learn
```

### 2. Run Dataset Cleaning Pipeline
```bash
python src/clean_dataset.py
```

### 3. Launch Web Application Server
```bash
python server.py
```
Or with uvicorn:
```bash
uvicorn server:app --reload --port 8000
```
Open **http://127.0.0.1:8000** in your browser!

---

## 📁 Repository Structure

```
.
├── server.py                      # FastAPI REST server & static file host
├── static/
│   ├── index.html                 # HTML5 single-page application
│   ├── style.css                  # Modern high-contrast Vanilla CSS theme
│   └── app.js                     # Modular JavaScript client logic
├── src/
│   ├── clean_dataset.py           # Dataset cleaning & spatial text pipeline
│   ├── document_processor.py      # Unified multi-format document reader
│   ├── models/
│   │   ├── trocr_extractor.py     # TrOCR & Hybrid OCR text extractor
│   │   ├── layoutlm_analyzer.py   # LayoutLMv3 spatial layout analyzer
│   │   └── donut_parser.py        # Donut Transformer structured IE model
│   └── agent/
│       ├── document_chat.py       # AI Document Chat & Q&A agent
│       ├── summarizer.py          # Executive & topic summarizer agent
│       ├── mcq_generator.py       # MCQ Quiz generator agent
│       └── keyword_highlighter.py # Visual bounding box highlighter
├── dataset/                       # Raw & Cleaned Parquets/CSVs
└── README.md                      # Project Documentation
```
