import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("faq_ingestion")

# -----------------------
# Constants
# -----------------------
REQUIRED_Q_FIELDS = {
    "fr", "en", "answer_fr", "answer_en", "subsidiary_fr", "subsidiary_en"
}
BATCH_SIZE = 16  # Embedding batch size

# -----------------------
# Helpers
# -----------------------
def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _validate_question(q: Dict, theme: str, idx: int):
    missing = REQUIRED_Q_FIELDS - q.keys()
    if missing:
        raise ValueError(f"[theme: {theme} / question idx {idx}] Missing fields: {missing}")

def load_faq_json(path: str) -> List[Dict]:
    """Load and flatten bilingual FAQ JSON into per-question items."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("FAQ JSON root must be a list of theme blocks")

    items = []
    for block in data:
        theme = block["theme"]
        questions = block.get("questions", [])
        if not isinstance(questions, list):
            raise ValueError(f"[theme: {theme}] questions must be a list")

        for idx, q in enumerate(questions):
            _validate_question(q, theme, idx)
            for lang in ("fr", "en"):
                # Main question
                items.append({
                    "lang": lang,
                    "theme": theme,
                    "question": q[lang].strip(),
                    "answer": q[f"answer_{lang}"].strip(),
                    "faq_id": _md5(f"{theme}|{lang}|{q[lang]}"),
                })
                # Subsidiary questions
                for sub in q.get(f"subsidiary_{lang}", []):
                    items.append({
                        "lang": lang,
                        "theme": theme,
                        "question": sub.strip(),
                        "answer": q[f"answer_{lang}"].strip(),
                        "faq_id": _md5(f"{theme}|{lang}|{sub}"),
                    })
    return items

def _build_payload(items: List[Dict]):
    ids, docs, metadatas = [], [], []
    for it in items:
        doc_text = f"Q: {it['question']}\nA: {it['answer']}"
        rid = f"{it['lang']}_{_md5(f'{it['lang']}|{it['theme']}|{it['question']}')}"
        ids.append(rid)
        docs.append(doc_text)
        metadatas.append(it)
    return ids, docs, metadatas

def _chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

# -----------------------
# Main ingestion function
# -----------------------
def ingest_faq_to_chroma(json_path: str, collection_path: str, collection_name: str):
    # Load FAQ
    logger.info("Loading FAQ JSON from %s", json_path)
    items = load_faq_json(json_path)
    logger.info("Expanded to %d items", len(items))

    # Deduplicate
    seen = set()
    dedup_items = []
    for it in items:
        if it["faq_id"] not in seen:
            dedup_items.append(it)
            seen.add(it["faq_id"])
    logger.info("Deduplicated to %d unique items", len(dedup_items))

    # Build payload
    ids, docs, metas = _build_payload(dedup_items)

    # Initialize embeddings and Chroma collection
    embeddings = OpenAIEmbeddings()
    collection_dir = Path(collection_path)
    collection_dir.mkdir(parents=True, exist_ok=True)
    coll = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(collection_dir)
    )

    # Batch insert
    total_batches = (len(ids) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx, idxs in enumerate(_chunk(list(range(len(ids))), BATCH_SIZE), 1):
        batch_ids = [ids[i] for i in idxs]
        batch_docs = [docs[i] for i in idxs]
        batch_metas = [metas[i] for i in idxs]

        try:
            coll.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas
            )
            logger.info("✅ Batch %d/%d inserted (%d items)", batch_idx, total_batches, len(batch_ids))
        except Exception as e:
            logger.exception("Failed to insert batch %d: %s", batch_idx, e)

    # Persist collection
    coll.persist()
    logger.info("✅ Ingestion complete into collection '%s' at '%s'", collection_name, collection_path)

    # Verify ingestion
    try:
        sample = coll.peek(limit=2)
        for i, doc in enumerate(sample["documents"], 1):
            meta = sample["metadatas"][i - 1]
            logger.info("[Sample %d] %s | meta=%s", i, doc[:120], meta)
    except Exception as e:
        logger.warning("Could not verify collection: %s", e)


if __name__ == "__main__":
    ingest_faq_to_chroma(
        json_path="faq_bilingual.json",
        collection_path="./chroma_db",
        collection_name="faq_collections_small"
    )
