# app/services/rag.py
import asyncio
import logging
import os
from typing import Dict, Any
from langdetect import detect, LangDetectException
from openai import AsyncOpenAI, OpenAIError
import chromadb
from app.core.config import settings  # holds CHROMA_DB_PATH, OPENAI_API_KEY, etc.

# -----------------------------
# Logging setup
# -----------------------------
logger = logging.getLogger("rag")
logger.setLevel(getattr(settings, "LOG_LEVEL", logging.INFO))

# -----------------------------
# Environment setup
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", getattr(settings, "OPENAI_API_KEY", ""))
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment or settings.")

# -----------------------------
# Initialize clients
# -----------------------------
client = AsyncOpenAI(api_key=OPENAI_API_KEY)
CHROMA_PATH = os.path.abspath(getattr(settings, "CHROMA_DB_PATH", "chroma_db"))
CHROMA_COLLECTION = getattr(settings, "CHROMA_COLLECTION", "faq_collections_small")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)

logger.info(f"[RAG] Using ChromaDB path: {CHROMA_PATH}")
logger.info(f"[RAG] Using Collection: {CHROMA_COLLECTION}")

# -----------------------------
# Constants / Configurable params
# -----------------------------
TOP_K = getattr(settings, "RAG_TOP_K", 3)
DISTANCE_THRESHOLD = getattr(settings, "RAG_DISTANCE_THRESHOLD", 0.35)  # smaller = better
CHROMA_QUERY_TIMEOUT = getattr(settings, "RAG_QUERY_TIMEOUT_SECS", 8.0)

# -----------------------------
# Main RAG Query Function
# -----------------------------
async def query_faq(question: str) -> Dict[str, Any]:
    """
    Query the FAQ vector database and generate a localized response using OpenAI.
    Returns a dictionary formatted for DeepChat: {"role": "bot", "text": str}
    """

    logger.info("[RAG] Collection name: %s, count: %s", collection.name, collection.count())

    # 1. Detect language
    try:
        detected_lang = detect(question)
    except LangDetectException:
        detected_lang = "en"
    logger.info(f"[RAG] Detected language: {detected_lang}")

    # 2. Generate embedding
    try:
        embedding_response = await client.embeddings.create(
            input=question,
            model=getattr(settings, "EMBEDDING_MODEL", "text-embedding-3-small"),
        )
        embed = embedding_response.data[0].embedding
        logger.debug(f"[RAG] Embedding length: {len(embed)}")
    except OpenAIError:
        logger.exception("[RAG] OpenAI Embedding failed")
        return {"role": "bot", "text": "Sorry, I couldn't process your question."}

    # 3. Query Chroma (run sync query off event loop)
    logger.info("[RAG] Going to query Chroma …")
    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(
                collection.query,
                query_embeddings=[embed],
                n_results=TOP_K,
                include=["documents", "metadatas", "distances"],
                where={"lang": detected_lang},  # language-aware filtering
            ),
            timeout=CHROMA_QUERY_TIMEOUT,
        )
        logger.info("[RAG] Chroma query finished, retrieved %d docs", len(results.get("documents", [[]])[0]))
    except asyncio.TimeoutError:
        logger.error("[RAG] ChromaDB query timed out")
        return {"role": "bot", "text": "Sorry, I couldn’t retrieve information in time."}
    except Exception:
        logger.exception("[RAG] ChromaDB query failed")
        return {"role": "bot", "text": "Sorry, I couldn't retrieve information."}

    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]

    if not docs:
        logger.warning("[RAG] No retrieval results")
        return {"role": "bot", "text": "Sorry, I couldn’t find an answer."}

    # 4. Sort by distance & filter
    triples = sorted(
        zip(docs, metas, dists),
        key=lambda t: t[2] if t[2] is not None else 99.0,
    )
    scored_docs = [
        (d, m, dist) for d, m, dist in triples if dist is not None and dist <= DISTANCE_THRESHOLD
    ]

    if not scored_docs:
        logger.info("[RAG] No document passed the distance threshold")
        return {"role": "bot", "text": "Sorry, I couldn’t find a reliable answer."}

    # 🔍 Debug log top retrieved docs
    for i, (d, m, dist) in enumerate(scored_docs[:3], 1):
        logger.debug(f"[RAG] Top {i} doc (distance={dist:.4f}): {d[:120]}... | meta={m}")

    # 5. Build context for LLM
    context_text = "\n".join([d for d, _, _ in scored_docs])
    system_prompt = (
        "You are a helpful healthcare assistant for question-answering tasks. "
        "Use the retrieved context to answer the question in the same language. "
        "If you don't know, say so. Keep the answer concise (max 3 sentences)."
    )
    user_prompt = f"Question: {question}\n\nFAQ Context:\n{context_text}"

    logger.debug(f"[RAG] SYSTEM PROMPT: {system_prompt}")
    logger.debug(f"[RAG] USER PROMPT: {user_prompt}")

    # 6. Query LLM
    try:
        response = await client.chat.completions.create(
            model=getattr(settings, "LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip()
        logger.info(f"[RAG] Answer generated: {answer}")
    except OpenAIError:
        logger.exception("[RAG] OpenAI Chat completion failed")
        answer = "Sorry, I couldn’t generate an answer."

    # 7. Return DeepChat-compatible response
    return {"role": "bot", "text": answer}
