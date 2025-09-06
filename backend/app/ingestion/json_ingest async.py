# app/ingestion/json_ingest.py
"""
Purpose:
1. Load a bilingual FAQ JSON.
2. Expand it into individual question-answer items (including subsidiary questions).
3. Deduplicate repeated questions.
4. Batch and embed the documents asynchronously.
5. Insert them into a vector database (Chroma).
6. Log the progress and verify ingestion.
"""

import json
import hashlib
import logging
import sys
from typing import Dict, List, Any
from tqdm import tqdm
import asyncio

from app.core.config import settings
from app.services.database import get_collection
from app.services.embeddings import embed_texts

# -----------------------------
# Logging setup
# -----------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ingestion")

# -----------------------------
# Constants
# -----------------------------
REQUIRED_Q_FIELDS = {"fr", "en", "answer_fr", "answer_en", "subsidiary_fr", "subsidiary_en"}

# -----------------------------
# Helpers
# -----------------------------
def _md5(s: str) -> str:
    """Generate an MD5 hash for a given string."""
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _validate_theme_block(block: Dict[str, Any], index: int):
    """Validate structure of a FAQ JSON theme block."""
    if "theme" not in block or "questions" not in block:
        raise ValueError(f"[theme idx {index}] Missing 'theme' or 'questions'")
    if not isinstance(block["questions"], list):
        raise ValueError(f"[theme idx {index}] 'questions' must be a list")
    for qi, q in enumerate(block["questions"]):
        missing = REQUIRED_Q_FIELDS - set(q.keys())
        if missing:
            raise ValueError(f"[theme idx {index} / question idx {qi}] Missing fields: {missing}")


def load_faq_json(path: str) -> List[Dict[str, Any]]:
    """
    Load a bilingual FAQ JSON file and expand it into per-question records.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Root JSON must be a list of theme blocks")

    for i, block in enumerate(data):
        _validate_theme_block(block, i)

    items: List[Dict[str, Any]] = []
    for block in data:
        theme = block["theme"]
        for q in block["questions"]:
            for lang in ("fr", "en"):
                # Main question
                items.append({
                    "lang": lang,
                    "theme": theme,
                    "question": q[lang],
                    "answer": q[f"answer_{lang}"],
                    "faq_id": _md5(f"{theme}|{lang}|{q[lang]}"),
                })
                # Subsidiary questions
                for sub in q.get(f"subsidiary_{lang}", []):
                    items.append({
                        "lang": lang,
                        "theme": theme,
                        "question": sub,
                        "answer": q[f"answer_{lang}"],
                        "faq_id": _md5(f"{theme}|{lang}|{sub}"),
                    })
    return items


def _build_payload(items: List[Dict[str, Any]]):
    """Prepare IDs, documents, and metadata for embedding and ingestion."""
    ids, documents, metadatas = [], [], []
    for it in items:
        lang = it["lang"]
        theme = it["theme"]
        q = it["question"].strip()
        a = it["answer"].strip()
        doc = f"Q: {q}\nA: {a}"
        rid = f"{lang}_{_md5(f'{lang}|{theme}|{q}')}"
        ids.append(rid)
        documents.append(doc)
        metadatas.append({
            "faq_id": it["faq_id"],
            "lang": lang,
            "theme": theme,
            "question": q,
            "answer": a
        })
    return ids, documents, metadatas


def _chunk(lst, size):
    """Split a list into smaller chunks."""
    for i in range(0, len(lst), size):
        yield lst[i:i+size]


# -----------------------------
# Verification
# -----------------------------
async def verify_ingestion(collection):
    """
    Verify that FAQ documents were successfully ingested into Chroma.
    """
    try:
        logger.info("Verifying ingestion by fetching 2 sample FAQ documents …")
        faq_sample = await collection.peek(limit=2)
        if faq_sample and "documents" in faq_sample:
            for i, doc in enumerate(faq_sample["documents"], 1):
                meta = faq_sample["metadatas"][i - 1]
                logger.info(
                    "[faq no. %d] lang=%s | doc=%s | meta=%s",
                    i, meta.get("lang"), doc, meta
                )
        else:
            logger.warning("No FAQ documents retrieved during verification")
    except Exception as e:
        logger.error("Verification failed: %s", e)


# -----------------------------
# Main async ingestion
# -----------------------------
async def run_ingestion():
    """Load, embed, and insert FAQ data into Chroma asynchronously."""
    logger.info("Loading FAQ JSON from %s", settings.JSON_FAQ_PATH)
    items = load_faq_json(settings.JSON_FAQ_PATH)
    logger.info("Expanded to %d items (including subsidiaries)", len(items))

    # Language distribution
    lang_count = {}
    for it in items:
        lang_count[it["lang"]] = lang_count.get(it["lang"], 0) + 1
    logger.info("Language distribution: %s", lang_count)

    ids, docs, metas = _build_payload(items)
    batch_size = getattr(settings, "EMBEDDING_BATCH_SIZE", 16)
    coll = get_collection()

    # Deduplicate
    seen = set()
    dedup_ids, dedup_docs, dedup_metas = [], [], []
    for i, rid in enumerate(ids):
        if rid in seen:
            continue
        seen.add(rid)
        dedup_ids.append(rid)
        dedup_docs.append(docs[i])
        dedup_metas.append(metas[i])

    logger.info("Deduplicated to %d unique items", len(dedup_ids))

    # Batch processing for embeddings & insertion
    total = len(dedup_ids)
    ingested_count = 0

    for idxs in tqdm(list(_chunk(list(range(total)), batch_size)), desc="Ingesting", unit="batch"):
        b_ids = [dedup_ids[i] for i in idxs]
        b_docs = [dedup_docs[i] for i in idxs]
        b_meta = [dedup_metas[i] for i in idxs]

        # Async embeddings
        try:
            embeddings = await embed_texts(b_docs)   # <- async call
            logger.debug("Embeddings ready for batch of %d docs", len(b_docs))
        except Exception as e:
            logger.exception("Embedding batch failed; skipping this batch: %s", e)
            continue

        # Insert into Chroma (upsert/add)
        try:
            if hasattr(coll, "upsert"):
                await coll.upsert(ids=b_ids, documents=b_docs, metadatas=b_meta, embeddings=embeddings)
            else:
                await coll.add(ids=b_ids, documents=b_docs, metadatas=b_meta, embeddings=embeddings)
            ingested_count += len(b_ids)
        except Exception as e:
            logger.exception("Chroma write failed for batch; skipping: %s", e)

    logger.info("Total items ingested: %d", ingested_count)

    # Final verification
    await verify_ingestion(coll)


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    import asyncio
    asyncio.run(run_ingestion())
