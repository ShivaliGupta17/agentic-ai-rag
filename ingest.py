import os
import sys
from pathlib import Path

import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

import vector_store as vs

load_dotenv()

PDF_URL = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
PDF_PATH = Path("data/Ebook-Agentic-AI.pdf")

CHUNK_SIZE_TOKENS = 700   
CHUNK_OVERLAP_TOKENS = 100

_tokenizer = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_tokenizer.encode(text))


def download_pdf() -> Path:
    """Download the source PDF if we don't already have a local copy."""
    PDF_PATH.parent.mkdir(exist_ok=True)
    if PDF_PATH.exists():
        print(f"[ingest] Found existing PDF at {PDF_PATH}, skipping download.")
        return PDF_PATH

    print(f"[ingest] Downloading source PDF from {PDF_URL} ...")
    response = requests.get(PDF_URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    PDF_PATH.write_bytes(response.content)
    print(f"[ingest] Saved PDF to {PDF_PATH} ({len(response.content) / 1024:.1f} KB)")
    return PDF_PATH


def extract_pages(pdf_path: Path) -> list[dict]:
    """Return a list of {page: int, text: str} for every non-empty page."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": page_number, "text": text})
    doc.close()
    print(f"[ingest] Extracted text from {len(pages)} non-empty pages.")
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Split each page's text into overlapping chunks, tagging every chunk
    with its source page number and a stable chunk id."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=_token_len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for i, chunk_text in enumerate(page_chunks):
            chunks.append(
                {
                    "chunk_id": f"page{page['page']:03d}-chunk{i:02d}",
                    "page": page["page"],
                    "text": chunk_text,
                }
            )
    print(f"[ingest] Produced {len(chunks)} chunks "
          f"(target size {CHUNK_SIZE_TOKENS} tokens, overlap {CHUNK_OVERLAP_TOKENS}).")
    return chunks


def main():
    backend = vs.active_backend()
    print(f"[ingest] Vector backend: {backend.upper()}"
          + ("" if backend == "pinecone" else " (no PINECONE_API_KEY found — using local FAISS)"))

    existing = vs.existing_vector_count()
    if existing > 0:
        print(f"[ingest] Vector store already contains {existing} vectors — "
              f"nothing to do. Delete the index / faiss_index/ folder to re-ingest.")
        return

    pdf_path = download_pdf()
    pages = extract_pages(pdf_path)
    if not pages:
        print("[ingest] ERROR: no extractable text found in the PDF.", file=sys.stderr)
        sys.exit(1)

    chunks = chunk_pages(pages)

    print(f"[ingest] Embedding {len(chunks)} chunks with all-MiniLM-L6-v2 (local)...")
    vectors = vs.embed_texts([c["text"] for c in chunks])

    print(f"[ingest] Upserting into {backend.upper()}...")
    vs.upsert_chunks(chunks, vectors)

    print(f"[ingest] Done. {len(chunks)} chunks stored in {backend.upper()}.")


if __name__ == "__main__":
    main()
