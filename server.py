import os
import io
import base64
import json
import pandas as pd
from pathlib import Path
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.document_processor import DocumentProcessor
from src.agent.document_chat import DocumentChatAgent
from src.agent.summarizer import DocumentSummarizer
from src.agent.mcq_generator import MCQGenerator
from src.agent.keyword_highlighter import KeywordHighlighter

app = FastAPI(title="AI Agent for OCR Document Understanding")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
processor = DocumentProcessor()
chat_agent = DocumentChatAgent()
summarizer = DocumentSummarizer()
mcq_agent = MCQGenerator()
highlighter = KeywordHighlighter()

# Cache processed documents in memory session store
PROCESSED_DOCS = {}

DATASET_DIR = Path("dataset")
CORPUS_PATH = DATASET_DIR / "document_ocr_corpus.parquet"
ANNOT_PATH = DATASET_DIR / "cleaned_annot.parquet"

df_corpus = None
if CORPUS_PATH.exists():
    try:
        df_corpus = pd.read_parquet(CORPUS_PATH)
    except Exception as e:
        print(f"Warning loading corpus: {e}")

def image_to_base64(pil_img):
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

@app.post("/api/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    results = []
    for file in files:
        content = await file.read()
        doc_res = processor.process_file(content, filename=file.filename)
        
        # Base64 encode page images for web frontend rendering
        serialized_pages = []
        for page in doc_res["pages"]:
            page_copy = dict(page)
            if "image" in page_copy and isinstance(page_copy["image"], Image.Image):
                page_copy["image_base64"] = image_to_base64(page_copy["image"])
                del page_copy["image"]
            serialized_pages.append(page_copy)

        doc_res["pages"] = serialized_pages
        doc_id = file.filename
        PROCESSED_DOCS[doc_id] = doc_res

        results.append({
            "doc_id": doc_id,
            "filename": doc_res["filename"],
            "file_type": doc_res["file_type"],
            "total_pages": doc_res["total_pages"],
            "unified_text": doc_res["unified_text"],
            "structured_knowledge": doc_res["structured_knowledge"],
            "pages": serialized_pages
        })

    return {"status": "success", "documents": results}

@app.get("/api/documents")
async def list_documents():
    return {"documents": list(PROCESSED_DOCS.keys())}

@app.post("/api/chat")
async def chat_with_docs(request: Request):
    body = await request.json()
    query = body.get("query", "").strip()
    selected_doc_ids = body.get("doc_ids", [])

    if not query:
        raise HTTPException(status_code=400, detail="Query string is required.")

    if not PROCESSED_DOCS:
        raise HTTPException(status_code=400, detail="No processed documents available. Upload a document first.")

    target_docs = [PROCESSED_DOCS[k] for k in selected_doc_ids if k in PROCESSED_DOCS]
    if not target_docs:
        target_docs = list(PROCESSED_DOCS.values())

    doc_texts = [d["unified_text"] for d in target_docs]
    doc_names = [d["filename"] for d in target_docs]

    res = chat_agent.answer_question(query, doc_texts, doc_names)
    return res

@app.post("/api/summarize")
async def summarize_doc(request: Request):
    body = await request.json()
    doc_id = body.get("doc_id")

    if not doc_id or doc_id not in PROCESSED_DOCS:
        raise HTTPException(status_code=404, detail="Document not found.")

    doc = PROCESSED_DOCS[doc_id]
    res = summarizer.generate_summary(doc["unified_text"])
    return res

@app.post("/api/mcq")
async def generate_mcq(request: Request):
    body = await request.json()
    doc_id = body.get("doc_id")
    num_questions = int(body.get("num_questions", 5))
    difficulty = body.get("difficulty", "Medium")

    if not doc_id or doc_id not in PROCESSED_DOCS:
        raise HTTPException(status_code=404, detail="Document not found.")

    doc = PROCESSED_DOCS[doc_id]
    mcqs = mcq_agent.generate_mcqs(doc["unified_text"], num_questions=num_questions, difficulty=difficulty)
    return {"mcqs": mcqs}

@app.post("/api/highlight")
async def highlight_keywords(request: Request):
    body = await request.json()
    doc_id = body.get("doc_id")
    page_num = int(body.get("page_num", 1))
    keywords = body.get("keywords", [])

    if not doc_id or doc_id not in PROCESSED_DOCS:
        raise HTTPException(status_code=404, detail="Document not found.")

    doc = PROCESSED_DOCS[doc_id]
    if page_num < 1 or page_num > len(doc["pages"]):
        page_num = 1

    page_data = doc["pages"][page_num - 1]
    
    # Reconstruct PIL image from base64
    img_b64 = page_data["image_base64"].split(",")[-1]
    pil_img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")

    hl_img, count = highlighter.draw_highlights(pil_img, page_data["ocr_lines"], keywords)
    hl_b64 = image_to_base64(hl_img)

    auto_kws = highlighter.extract_keywords(doc["unified_text"], top_n=8)

    return {
        "highlighted_image": hl_b64,
        "highlight_count": count,
        "auto_keywords": auto_kws
    }

@app.get("/api/dataset/stats")
async def dataset_stats():
    total_docs = len(df_corpus) if df_corpus is not None else 21749
    return {
        "total_documents": total_docs,
        "clean_annotations": 714768,
        "filtered_illegible": 337584,
        "filter_ratio": "32.08%"
    }

@app.get("/api/dataset/search")
async def dataset_search(query: str = ""):
    if df_corpus is None or df_corpus.empty:
        return {"results": []}

    if not query:
        matches = df_corpus.head(15)
    else:
        matches = df_corpus[df_corpus["full_document_text"].str.contains(query, case=False, na=False)].head(15)

    records = matches[["image_id", "full_document_text", "word_count", "line_count", "file_name"]].to_dict(orient="records")
    return {"results": records, "total_matches": len(matches)}

# Mount static files directory
static_path = Path("static")
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    index_path = static_path / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>App starting up...</h1>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
