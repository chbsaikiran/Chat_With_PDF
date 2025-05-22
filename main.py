import trafilatura
import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin
import google.generativeai as genai
import os
from dotenv import load_dotenv
import numpy as np
import faiss
import pymupdf4llm
from pathlib import Path
import hashlib
import re
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional
#from models import FilePathInput
ROOT = Path(__file__).parent.resolve()
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configure the model
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

# List available models
# print("Available Models:")
# for m in genai.list_models():
#     print(f"- {m.name}")

# Initialize the model
model = genai.GenerativeModel("gemini-2.0-flash",
                            generation_config=generation_config)

@dataclass
class UrlInput:
    url: str

@dataclass
class MarkItDown:
    def convert(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            return type('obj', (object,), {'text_content': f.read()})

def get_embedding(text: str) -> np.ndarray:
    """Get embedding for text using Gemini embedding model."""
    try:
        embedding_model = genai.embed_content(
            model="models/embedding-001",  # This is the correct model ID
            content=text,
            task_type="retrieval_document"
        )
        return np.array(embedding_model['embedding'])
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return np.zeros(768)

def caption_image(image_path: str) -> str:
    """Generate caption for image using Gemini model."""
    try:
        response = model.generate_content(f"Describe this image: {image_path}")
        return response.text
    except Exception as e:
        print(f"Error generating caption: {e}")
        return "Image description not available"

def replace_images_with_captions(markdown):
    def replace(match):
        alt, src = match.group(1), match.group(2)
        try:
            caption = caption_image(src)
            # Attempt to delete only if local and file exists
            if not src.startswith("http"):
                img_path = Path(__file__).parent / "documents" / src
                if img_path.exists():
                    img_path.unlink()
                    print(f"🗑️ Deleted image after captioning: {img_path}")
            return f"**Image:** {caption}"
        except Exception as e:
            print(f"Image deletion failed: {e}")
            return f"[Image could not be processed: {src}]"

    return re.sub(r'!\[(.*?)\]\((.*?)\)', replace, markdown)

def extract_pdf(file_path):
    """Convert PDF to markdown. Usage: input={"input": {"file_path": "documents/sample.pdf"} } result = await mcp.call_tool('extract_pdf', input)"""


    if not os.path.exists(file_path):
        return f"File not found: {file_path}"

    ROOT = Path(__file__).parent.resolve()
    global_image_dir = ROOT / "documents" / "images"
    global_image_dir.mkdir(parents=True, exist_ok=True)

    # Actual markdown with relative image paths
    markdown = pymupdf4llm.to_markdown(
        file_path,
        write_images=True,
        image_path=str(global_image_dir)
    )

    # Re-point image links in the markdown
    markdown = re.sub(
        r'!\[\]\((.*?/images/)([^)]+)\)',
        r'![](images/\2)',
        markdown.replace("\\", "/")
    )

    markdown = replace_images_with_captions(markdown)
    return markdown

def semantic_merge(text):
    """Splits text semantically using LLM: detects second topic and reuses leftover intelligently."""
    WORD_LIMIT = 512
    words = text.split()
    i = 0
    final_chunks = []

    while i < len(words):
        chunk_words = words[i:i + WORD_LIMIT]
        chunk_text = " ".join(chunk_words).strip()

        prompt = f"""
You are a markdown document segmenter.

Here is a portion of a markdown document:

---
{chunk_text}
---

If this chunk clearly contains **more than one distinct topic or section**, reply ONLY with the **second part**, starting from the first sentence or heading of the new topic.

If it's only one topic, reply with NOTHING.

Keep markdown formatting intact.
"""

        try:
            response = model.generate_content(prompt)
            reply = response.text.strip()

            if reply:
                split_point = chunk_text.find(reply)
                if split_point != -1:
                    first_part = chunk_text[:split_point].strip()
                    second_part = reply.strip()

                    final_chunks.append(first_part)

                    leftover_words = second_part.split()
                    words = leftover_words + words[i + WORD_LIMIT:]
                    i = 0
                    continue
                else:
                    final_chunks.append(chunk_text)
            else:
                final_chunks.append(chunk_text)

        except Exception as e:
            print(f"[ERROR] Semantic chunking LLM error: {e}")
            final_chunks.append(chunk_text)

        i += WORD_LIMIT

    return final_chunks

def process_documents():
    """Process documents and create FAISS index using unified multimodal strategy."""
    print("Indexing documents with unified RAG pipeline...")
    ROOT = Path(__file__).parent.resolve()
    DOC_PATH = ROOT / "documents"
    INDEX_CACHE = ROOT / "faiss_index"
    INDEX_CACHE.mkdir(exist_ok=True)
    INDEX_FILE = INDEX_CACHE / "index.bin"
    METADATA_FILE = INDEX_CACHE / "metadata.json"
    CACHE_FILE = INDEX_CACHE / "doc_index_cache.json"

    def file_hash(path):
        return hashlib.md5(Path(path).read_bytes()).hexdigest()

    CACHE_META = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    metadata = json.loads(METADATA_FILE.read_text()) if METADATA_FILE.exists() else []
    index = faiss.read_index(str(INDEX_FILE)) if INDEX_FILE.exists() else None

    for file in DOC_PATH.glob("*.*"):
        fhash = file_hash(file)
        if file.name in CACHE_META and CACHE_META[file.name] == fhash:
            print(f"Skipping unchanged file: {file.name}")
            continue

        print(f"Processing: {file.name}")
        try:
            ext = file.suffix.lower()
            markdown = ""

            if ext == ".pdf":
                print(f"Using MuPDF4LLM to extract {file.name}")
                markdown = extract_pdf(str(file))

            elif ext in [".html", ".htm", ".url"]:
                print(f"Using Trafilatura to extract {file.name}")
                markdown = extract_webpage(UrlInput(url=file.read_text().strip())).markdown

            else:
                # Fallback to MarkItDown for other formats
                converter = MarkItDown()
                print(f"Using MarkItDown fallback for {file.name}")
                markdown = converter.convert(str(file)).text_content

            if not markdown.strip():
                print(f"No content extracted from {file.name}")
                continue

            if len(markdown.split()) < 10:
                print(f"Content too short for semantic merge in {file.name} → Skipping chunking.")
                chunks = [markdown.strip()]
            else:
                print(f"Running semantic merge on {file.name} with {len(markdown.split())} words")
                chunks = semantic_merge(markdown)


            embeddings_for_file = []
            new_metadata = []
            for i, chunk in enumerate(tqdm(chunks, desc=f"Embedding {file.name}")):
                embedding = get_embedding(chunk)
                embeddings_for_file.append(embedding)
                new_metadata.append({
                    "doc": file.name,
                    "chunk": chunk,
                    "chunk_id": f"{file.stem}_{i}"
                })

            if embeddings_for_file:
                if index is None:
                    dim = len(embeddings_for_file[0])
                    index = faiss.IndexFlatL2(dim)
                index.add(np.stack(embeddings_for_file))
                metadata.extend(new_metadata)
                CACHE_META[file.name] = fhash

                # ✅ Immediately save index and metadata
                CACHE_FILE.write_text(json.dumps(CACHE_META, indent=2))
                METADATA_FILE.write_text(json.dumps(metadata, indent=2))
                faiss.write_index(index, str(INDEX_FILE))
                print(f"Saved FAISS index and metadata after processing {file.name}")

        except Exception as e:
            print(f"Failed to process {file.name}: {e}")

def ensure_faiss_ready():
    from pathlib import Path
    index_path = ROOT / "faiss_index" / "index.bin"
    meta_path = ROOT / "faiss_index" / "metadata.json"
    if not (index_path.exists() and meta_path.exists()):
        print("Index not found — running process_documents()...")
        process_documents()
    else:
        print("Index already exists. Skipping regeneration.")

def search_stored_documents(query):
    """Search documents to get relevant extracts. Usage: input={"input": {"query": "your query"}} result = await mcp.call_tool('search_stored_documents', input)"""

    ensure_faiss_ready()
    print(f"Query: {query}")
    try:
        index = faiss.read_index(str(ROOT / "faiss_index" / "index.bin"))
        metadata = json.loads((ROOT / "faiss_index" / "metadata.json").read_text())
        query_vec = get_embedding(query ).reshape(1, -1)
        D, I = index.search(query_vec, k=5)
        results = []
        for idx in I[0]:
            data = metadata[idx]
            results.append(f"{data['chunk']}\n[Source: {data['doc']}, ID: {data['chunk_id']}]")
        return results
    except Exception as e:
        return [f"ERROR: Failed to search: {str(e)}"]

def main():
    query = "what are the energy efficient practices for DLF?? list top 5 practices"
    results = search_stored_documents(query)
    prompt = f"""
    You are a helpful assistant that can answer questions about the documents in the following list:
    {results}
    Please answer the following question: {query}
    """
    response = model.generate_content(prompt)
    reply = response.text.strip()
    print(reply)

if __name__ == "__main__":
    main()

