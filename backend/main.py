# backend/main_doc2vec.py
import os, shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
from typing import List
from backend.ingest import pdf_to_documents, ensure_dir  # your previous chunking functions
from qdrant_client import QdrantClient
from backend.doc2vec_utils import prepare_tagged_documents, train_doc2vec, save_model, load_model, infer_vector, simple_tokenize
from backend.qdrant_upsert import ensure_qdrant_collection, upsert_chunks_to_qdrant
from tqdm import tqdm
import hashlib

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL").replace("https://", "").replace("http://", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "pdf_docs_doc2vec")
STORAGE_DIR = os.getenv("STORAGE_DIR", "./pdf_store")
DOC2VEC_MODEL_PATH = os.getenv("DOC2VEC_MODEL_PATH", "./models/doc2vec.model")

ensure_dir(STORAGE_DIR)
ensure_dir(os.path.dirname(DOC2VEC_MODEL_PATH) or "./models")

app = FastAPI(title="PDF Search (FastAPI + Qdrant + Doc2Vec)")

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)

# If model exists, load it; else None (we'll train on first ingestion)
if os.path.exists(DOC2VEC_MODEL_PATH):
    d2v_model = load_model(DOC2VEC_MODEL_PATH)
else:
    d2v_model = None

@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """
    Upload multiple PDFs; save and then ingest (train/extend Doc2Vec and upsert chunks).
    This implementation does a naive train-on-all approach when a new model is not present.
    For production, do incremental training or background jobs.
    """
    uploaded = []
    for upload in files:
        filename = upload.filename
        dest_path = os.path.join(STORAGE_DIR, filename)
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(upload.file, f)
        uploaded.append({"filename": filename, "path": dest_path})

    # Build chunk list for training/embedding
    chunk_records = []  # each: {"id": uid, "text": text, "metadata": {...}}
    for entry in uploaded:
        docs = pdf_to_documents(entry["path"], entry["filename"], chunk_size_chars=1000, overlap_ratio=0.2)
        for d in docs:
            # generate deterministic id per chunk so duplicates can be detected later
            # e.g., sha256(pdf_name|page|start|end)
            md = d.metadata
            uid_source = f"{md['pdf_name']}|{md['page_num']}|{md['start_line']}|{md['end_line']}"
            uid = hashlib.sha256(uid_source.encode("utf-8")).hexdigest()
            chunk_records.append({"id": uid, "text": d.page_content, "metadata": md})

    if not chunk_records:
        return {"status": "no_chunks", "count": 0}

    # Train model if not exists; else update (incremental)
    tagged = prepare_tagged_documents([{"id": rec["id"], "text": rec["text"]} for rec in chunk_records])
    global d2v_model
    if d2v_model is None:
        d2v_model = train_doc2vec(tagged, vector_size=256, window=8, min_count=1, epochs=60, dm=1)
    else:
        # incremental update: build_vocab(update=True) then train on new docs
        d2v_model.build_vocab(tagged, update=True)
        d2v_model.train(tagged, total_examples=len(tagged), epochs=20)

    save_model(d2v_model, DOC2VEC_MODEL_PATH)

    # embed chunks
    embeddings = []
    metadatas = []
    for rec in chunk_records:
        vec = infer_vector(d2v_model, rec["text"], steps=30).tolist()
        embeddings.append(vec)
        # payload metadata: include pdf_name, page_num, start_line, end_line
        md = rec["metadata"]
        metadatas.append({
            "pdf_name": md.get("pdf_name"),
            "page_num": md.get("page_num"),
            "start_line": md.get("start_line"),
            "end_line": md.get("end_line"),
            "chunk_id": rec["id"]
        })

    # ensure collection exists
    ensure_qdrant_collection(qdrant_client, QDRANT_COLLECTION, vector_size=len(embeddings[0]))
    # upsert (in batches if large)
    upsert_chunks_to_qdrant(qdrant_client, QDRANT_COLLECTION, embeddings, metadatas)

    return {"status": "ok", "ingested_chunks": len(embeddings)}

@app.get("/search")
def search(q: str, top_k: int = 10):
    if d2v_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not trained yet. Ingest some PDFs first.")
    # compute query vector
    qvec = infer_vector(d2v_model, q, steps=30).tolist()
    # search qdrant
    search_result = qdrant_client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=qvec,
        limit=top_k,
        with_payload=True,
        with_vector=False
    )
    results = []
    for hit in search_result:
        payload = hit.payload or {}
        results.append({
            "pdf_name": payload.get("pdf_name"),
            "page_num": payload.get("page_num"),
            "start_line": payload.get("start_line"),
            "end_line": payload.get("end_line"),
            "chunk_id": payload.get("chunk_id"),
            "score": hit.score
        })
    # first occurrence per pdf
    first_occ = {}
    for r in results:
        pdf = r["pdf_name"]
        key = (r.get("page_num", 999999), r.get("start_line", 999999))
        if pdf not in first_occ or key < (first_occ[pdf]["page_num"], first_occ[pdf]["start_line"]):
            first_occ[pdf] = {"page_num": r["page_num"], "start_line": r["start_line"], "end_line": r["end_line"], "score": r["score"]}

    return {"query": q, "hits": results, "first_occurrence_by_pdf": first_occ}


# # backend/main.py
# import os
# import shutil
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import FileResponse
# from dotenv import load_dotenv
# from typing import List
# from backend.ingest import pdf_to_documents, ensure_dir
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Qdrant
# from qdrant_client import QdrantClient
# from qdrant_client.http import models as rest
# from tqdm import tqdm

# load_dotenv()

# QDRANT_URL = os.getenv("QDRANT_URL")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "pdf_docs")
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# STORAGE_DIR = os.getenv("STORAGE_DIR", "./pdf_store")

# ensure_dir(STORAGE_DIR)

# app = FastAPI(title="PDF Search (FastAPI + Qdrant + LangChain)")

# # create qdrant client
# if not QDRANT_URL or not QDRANT_API_KEY:
#     raise RuntimeError("Set QDRANT_URL and QDRANT_API_KEY in env")

# qdrant_client = QdrantClient(
#     url=QDRANT_URL.replace("https://", "").replace("http://", ""),  # qdrant-client expects host only for cloud
#     api_key=QDRANT_API_KEY,
#     prefer_grpc=False,
# )

# # embeddings via HuggingFace (sentence-transformers)
# hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# # Create collection if not exists (collection config)
# # NOTE: LangChain Qdrant.from_documents will create the collection if needed, but we set a schema here for clarity.
# try:
#     existing = qdrant_client.get_collections()
#     coll_names = [c.name for c in existing.collections]
#     if QDRANT_COLLECTION not in coll_names:
#         qdrant_client.recreate_collection(
#             collection_name=QDRANT_COLLECTION,
#             vectors_config=rest.VectorParams(size=hf_embeddings.embed_query("a").shape[0], distance=rest.Distance.COSINE)
#         )
# except Exception as exc:
#     # Some Qdrant cloud endpoints may require different handling; ignore if already exists
#     pass

# # helper: Upload and ingest multiple PDFs
# @app.post("/upload")
# async def upload_pdfs(files: List[UploadFile] = File(...)):
#     """
#     Upload multiple PDF files; these will be saved to STORAGE_DIR and ingested to Qdrant with metadata.
#     """
#     uploaded = []
#     for upload in files:
#         filename = upload.filename
#         dest_path = os.path.join(STORAGE_DIR, filename)
#         # save file
#         with open(dest_path, "wb") as f:
#             shutil.copyfileobj(upload.file, f)
#         uploaded.append({"filename": filename, "path": dest_path})

#     # ingest each file sequentially (for many files, run this in background worker)
#     total_docs = 0
#     for entry in tqdm(uploaded, desc="Ingesting PDFs"):
#         docs = pdf_to_documents(entry["path"], entry["filename"], chunk_size_chars=1000, overlap_ratio=0.2)
#         if not docs:
#             continue
#         # push to Qdrant via LangChain wrapper
#         # NOTE: to avoid duplicate insertion, you may want to use upsert ids; LangChain will auto-generate ids
#         qdrant_store = Qdrant.from_documents(
#             documents=docs,
#             embedding=hf_embeddings,
#             url=QDRANT_URL,
#             prefer_grpc=False,
#             api_key=QDRANT_API_KEY,
#             collection_name=QDRANT_COLLECTION
#         )
#         total_docs += len(docs)

#     return {"status": "ok", "ingested_files": [u["filename"] for u in uploaded], "total_chunks_indexed": total_docs}


# # search endpoint
# @app.get("/search")
# def search(q: str, top_k: int = 10):
#     """
#     Perform semantic search on Qdrant via LangChain retriever. Returns list of hits with metadata.
#     Also computes 'first occurrence per PDF' by earliest page_num and start_line among hits.
#     """
#     # recreate the store wrapper (LangChain's Qdrant requires it)
#     qdrant_store = Qdrant(
#         client=qdrant_client,
#         collection_name=QDRANT_COLLECTION,
#         embeddings=hf_embeddings
#     )
#     # use retriever
#     retriever = qdrant_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
#     docs = retriever.get_relevant_documents(q)

#     # format results
#     results = []
#     for d in docs:
#         md = d.metadata or {}
#         snippet = d.page_content[:800].replace("\n", " ")
#         results.append({
#             "pdf_name": md.get("pdf_name"),
#             "page_num": md.get("page_num"),
#             "start_line": md.get("start_line"),
#             "end_line": md.get("end_line"),
#             "chunk_index": md.get("chunk_index"),
#             "snippet": snippet
#         })

#     # compute first occurrence per PDF
#     first_occ = {}
#     for r in results:
#         pdf = r["pdf_name"]
#         # choose earliest by page_num then start_line
#         key = (r["page_num"] or 999999, r["start_line"] or 999999)
#         if pdf not in first_occ or key < (first_occ[pdf]["page_num"], first_occ[pdf]["start_line"]):
#             first_occ[pdf] = {
#                 "page_num": r["page_num"],
#                 "start_line": r["start_line"],
#                 "end_line": r["end_line"],
#                 "snippet": r["snippet"]
#             }

#     return {"query": q, "hits": results, "first_occurrence_by_pdf": first_occ}


# # endpoint to download stored pdf
# @app.get("/download/{pdf_name}")
# def download(pdf_name: str):
#     path = os.path.join(STORAGE_DIR, pdf_name)
#     if not os.path.exists(path):
#         raise HTTPException(status_code=404, detail="File not found")
#     return FileResponse(path, media_type="application/pdf", filename=pdf_name)
