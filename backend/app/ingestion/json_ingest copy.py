# Purpose:
# 1. Load a bilingual FAQ JSON.
# 2. Expand it into individual question-answer items (including subsidiary questions).
# 3. Deduplicate repeated questions.
# 4. Batch and embed the documents.
# 5. Insert them into a vector database (like Chroma).
# 6. Log the progress and verify ingestion.


import json
import hashlib
import logging
from typing import Dict, List, Any
from tqdm import tqdm
import sys

from app.core.config import settings
from app.services.database import get_collection, test_chroma_collection
from app.services.embeddings import embed_texts

# -----------------------------
# Logging setup
# %(asctime)s → Timestamp (when log happened).
# %(levelname)-8s → Log severity (INFO, DEBUG, ERROR). -8s means align left and reserve 8 spaces.
# %(name)s → Name of the logger (here: "ingestion").
# %(message)s → Actual log text.
# -----------------------------


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
     # logs print to the console add FileHandler("app.log") if you want logs stored
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Creates a named logger called ingestion
logger = logging.getLogger("ingestion")


# -----------------------------
# Constants
# -----------------------------
REQUIRED_Q_FIELDS = {
    "fr", "en", "answer_fr", "answer_en", "subsidiary_fr", "subsidiary_en"
}

# -----------------------------
# Helpers
# -----------------------------
def _md5(s: str) -> str:
    """
    Generate an MD5 hash for a given string.
    Args:
        s (str): The input string.
    Returns:
        str: A hexadecimal string representing the MD5 hash.
    """
    # 1. Encode the string into bytes using UTF-8 encoding
    # (hash functions always work on bytes, not Python str directly)
    # 2. Compute the MD5 hash of the byte string
    # 3. Convert the hash object into a hexadecimal string (readable format)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _validate_theme_block(block: Dict[str, Any], index: int):
    """
    Validate a FAQ JSON file structure.

    Args:
        block (Dict[str, Any]): The theme block to validate.
        index (int): Position of this block in the outer list (for error reporting).

    Raises:
        ValueError: If required keys or fields are missing, or if 'questions' is not a list.
    """
    #  Check that the block has the required top-level keys
    # Every theme block must contain:
    #   - "theme": the name of the theme (e.g., "Nutrition")
    #   - "questions": a list of questions belonging to that theme
    if "theme" not in block or "questions" not in block:
        raise ValueError(f"[theme idx {index}] Missing 'theme' or 'questions'")
    # Ensure "questions" is actually a list (not a dict or string)
    if not isinstance(block["questions"], list):
        raise ValueError(f"[theme idx {index}] 'questions' must be a list")
    # Validate each question inside the list
    for qi, q in enumerate(block["questions"]):
        # q.keys() gives you all the keys of the q dictionary.
        # Take all required fields, and subtract the ones that already exist in q.
        # If nothing is missing: result is an empty set (set()).
        # If some fields are missing: result is a set containing only those missing ones.
        missing = REQUIRED_Q_FIELDS - set(q.keys()) 
         # If the question is missing any required field, raise an error
        if missing:
            raise ValueError(f"[theme idx {index} / question idx {qi}] Missing fields: {missing}")

def load_faq_json(path: str) -> List[Dict[str, Any]]:
    """
    Load a bilingual FAQ JSON file and expand it into per-question records.

    The input JSON is expected to be a list of "theme blocks".
    Each block has:
      - "theme" (str): category of questions (e.g., "Appointments")
      - "questions" (list of dicts): bilingual questions + answers
        Each question dict should include:
            fr, en               -> main question text (French & English)
            answer_fr, answer_en -> answers in both languages
            subsidiary_fr, subsidiary_en -> optional lists of alternative phrasings

    The function validates the structure, then flattens it into a list of
    per-question records, one record for each language and each variant.
    """
    # Step 1: Load JSON file into Python object (list/dict)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Step 2: Validate top-level structure
    # Expecting a list of theme blocks, otherwise raise error
    if not isinstance(data, list):
        raise ValueError("Root JSON must be a list of theme blocks")
    
    # Step 3: Validate each theme block (theme, questions and its fields)
    for i, block in enumerate(data):
        _validate_theme_block(block, i)
    
    # Step 4: Prepare a container for flattened items
    items: List[Dict[str, Any]] = []

    # Step 5: Iterate over theme blocks
    for block in data:
        theme = block["theme"]

        # Step 6: Iterate over each question in the block
        for q in block["questions"]:

            # Step 7: Process both languages (French + English)
            for lang in ("fr", "en"):

                # Add the main question record
                items.append({
                    "lang": lang,                     # which language (fr/en)
                    "theme": theme,                   # theme/category
                    "question": q[lang],              # main question text
                    "answer": q[f"answer_{lang}"],    # corresponding answer
                    "faq_id": _md5(f"{theme}|{lang}|{q[lang]}"),  
                    # unique ID generated by hashing theme+lang+question
                })

                # Step 8: Handle subsidiary (alternative) phrasings
                # Example: "How can I make an appointment?" vs.
                #          "Can I book an appointment online?"
                for sub in q.get(f"subsidiary_{lang}", []):
                    items.append({
                        "lang": lang,
                        "theme": theme,
                        "question": sub,                  # alternative question
                        "answer": q[f"answer_{lang}"],    # same answer
                        "faq_id": _md5(f"{theme}|{lang}|{sub}"),
                    })

    # Step 9: Return the flattened list of records
    return items

def _build_payload(items: List[Dict[str, Any]]):
    """
    Build payloads for ingestion into a vector database or LangChain workflow.

    Args:
        items (List[Dict[str, Any]]): List of FAQ records, each containing
            'lang', 'theme', 'question', 'answer', and 'faq_id'.

    Returns:
        Tuple[List[str], List[str], List[Dict[str, Any]]]:
            - ids: unique record IDs for each FAQ item
            - documents: formatted Q&A text for embeddings
            - metadatas: metadata dictionaries for each record
    """

    # Lists to hold final payload components
    ids, documents, metadatas = [], [], []

    # Iterate over each FAQ record
    for it in items:
        lang = it["lang"]                # Language of the question (fr/en)
        theme = it["theme"]              # Theme or category
        q = it["question"].strip()       # Clean question text
        a = it["answer"].strip()         # Clean answer text

        # Combine question and answer into a single string for embeddings
        doc = f"Q: {q}\nA: {a}"

        # Generate a unique record ID: language + md5 hash of (lang|theme|question)
        rid = f"{lang}_{_md5(f'{lang}|{theme}|{q}')}"

        # Append ID, document text, and metadata to respective lists
        ids.append(rid)
        documents.append(doc)
        metadatas.append({
            "faq_id": it["faq_id"],  # Original FAQ ID from load_faq_json
            "lang": lang,            # Language
            "theme": theme,          # Theme/category
            "question": q,           # Question text
            "answer": a              # Answer text
        })

    # Return all three lists for ingestion (vector DB, LangChain, etc.)
    return ids, documents, metadatas


def _chunk(lst, size):
    """
    Split a list into smaller chunks of a given size.

    Args:
        lst (list): The input list to split.
        size (int): Maximum size of each chunk.

    Yields:
        list: Sub-lists of the original list, each up to 'size' elements.
    
    Example:
        list(_chunk([1,2,3,4,5], 2)) 
        -> [[1,2], [3,4], [5]]
    """
    # Loop over the list with a step of 'size'
    for i in range(0, len(lst), size):
        # Yield a slice from index i to i+size (handles last chunk automatically)
        yield lst[i:i+size]

# -----------------------------
# Verification
# -----------------------------
def verify_ingestion(collection):
    """
    Verify that FAQ documents were successfully ingested into a vector database or collection.

    Args:
        collection: The collection object (e.g., Chroma collection) to query.
    """

    try:
        # Log that verification is starting
        logger.info("Verifying ingestion by fetching 2 sample FAQ documents …")

        # Peek retrieves up to `limit` documents from the collection without removing them
        faq_sample = collection.peek(limit=2)

        # Check if documents were returned and the expected key exists
        if faq_sample and "documents" in faq_sample:
            # Iterate over the sample documents
            for i, doc in enumerate(faq_sample["documents"], 1):
                # Corresponding metadata for the document
                meta = faq_sample["metadatas"][i-1]

                # Log details: faq number, language, document text, and metadata
                logger.info(
                    "[faq no. %d] lang=%s | doc=%s | meta=%s",
                    i, meta.get("lang"), doc, meta
                )
        else:
            # No documents were returned from the collection
            logger.warning("No FAQ documents retrieved during verification")

    except Exception as e:
        # Catch any error during the verification process and log it
        logger.error("Verification failed: %s", e)

def run_ingestion():
    logger.info("Loading FAQ JSON from %s", settings.JSON_FAQ_PATH)
    # Reads our nested FAQ JSON and flattens it
    items = load_faq_json(settings.JSON_FAQ_PATH)
    logger.info("Expanded to %d items (including subsidiaries)", len(items))

    # Language distribution:
    # This counts how many items are in French vs. English.
    # Useful for verifying that our  bilingual data is balanced or correctly loaded.
    lang_count = {}
    for it in items:
        lang_count[it["lang"]] = lang_count.get(it["lang"], 0) + 1
    logger.info("Language distribution: %s", lang_count)

    ids, docs, metas = _build_payload(items)
    batch_size = settings.EMBEDDING_BATCH_SIZE
    coll = get_collection()

    # Deduplicate - Prevents duplicate questions from being inserted into the vector DB.
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
    total_batches = (len(dedup_ids) + batch_size - 1) // batch_size
    ingested_count = 0

    for batch_idx, idxs in enumerate(_chunk(list(range(len(dedup_ids))), batch_size), 1):
        b_ids = [dedup_ids[i] for i in idxs]
        b_docs = [dedup_docs[i] for i in idxs]
        b_meta = [dedup_metas[i] for i in idxs]

        # logger.info("batch ids  %s ", b_ids)
        # logger.info("batch doc  %s ", b_docs)
        # logger.info("batch meta  %s ", b_meta)
        # converts batch text into vector embeddings for semantic search.
        try:
            embeddings = embed_texts(b_docs)
            # logger.info("embeddings %s", embeddings)
            logger.debug("Batch %d embeddings shape: %d vectors for %d docs",
                     batch_idx, len(embeddings), len(b_docs))
        except Exception as e:
            # tqdm.write(f"[ERROR] Embedding batch {batch_idx}/{total_batches} failed: {e}")
            logger.exception("Embedding batch %d failed: %s", batch_idx, e)
            continue
        
        
        # Insert embeddings into collection
        # Handles two possible DB methods: upsert or add
        # upsert → updates existing IDs, avoid adding the same documents every time
        # add → just inserts new items.
        # Logs success/failure for each batch.
        
        try:
            logger.info("Going to insert batch %d into collection …", batch_idx) 
            # List all attributes/methods
            # print(dir(coll))

            # Check if it has upsert/add
            # print("has upsert:", hasattr(coll, "upsert"))
            # print("has add   :", hasattr(coll, "add"))

            # Count of documents
            print("Total documents in collection:", coll.count())
            if hasattr(coll, "upsert"):
                try:
                    logger.info("Calling coll.upsert ...")
                    logger.info("Embedding count: %d, Documents count: %d", len(embeddings), len(b_docs))
                    coll.upsert(
                        ids=b_ids,
                        documents=b_docs,
                        metadatas=b_meta,
                        embeddings=embeddings
                    )
                    logger.info("✅ Updated embeddings to collections")  # Guaranteed to print if no exception
                except Exception as e:
                    logger.exception("coll.upsert failed: %s", e)
            else:
                logger.info("Calling coll.add ...")
                coll.add(ids=b_ids, documents=b_docs, metadatas=b_meta, embeddings=embeddings)

            logger.info("✅ Batch %d inserted (%d items)", batch_idx, len(b_ids))
            ingested_count += len(b_ids)
            # Persist collection
            logger.info("✅ Ingestion complete into collection ");
            # tqdm.write(f"[INFO] Batch {batch_idx}/{total_batches} ingested: {len(b_ids)} items (running total: {ingested_count})")
        except Exception as e:
            # tqdm.write(f"[ERROR] Chroma write failed for batch {batch_idx}/{total_batches}: {e}")
            logger.exception("Chroma write failed for batch %d", batch_idx)
    # verify_ingestion(coll) # Peek 2 sample docs

    # Final verification
    try:
        total_count = coll.count()
        logger.info("✅ Ingestion complete. Collection '%s' contains %d items.",
                    getattr(coll, "name", "faqs"), total_count)

        # Peek 2 sample docs
        sample = coll.peek(limit=2)
        if sample and "documents" in sample:
            for i, doc in enumerate(sample["documents"], 1):
                meta = sample["metadatas"][i - 1] if "metadatas" in sample else {}
                logger.info("[Sample %d] %s | meta=%s", i, doc[:120], meta)
    except Exception as e:
        logger.warning("Could not verify collection: %s", e)

    logger.info("All batches processed. Total items ingested: %d", ingested_count)

if __name__ == "__main__":
    # run_ingestion()
    test_chroma_collection(collection_name="faq_collections_small",
    limit=3
)


    
