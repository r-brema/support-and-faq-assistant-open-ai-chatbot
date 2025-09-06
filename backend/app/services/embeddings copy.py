import logging
from openai import OpenAI
from app.core.config import settings

logger = logging.getLogger("embeddings")

# Initialize OpenAI client
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# -------------------------------
# Embedding utilities
# -------------------------------

def embed_text(text: str):
    """
    Embed a single piece of text using OpenAI's embedding model.
    Returns a vector (list of floats).
    """
    try:
        # Create embedding request
        resp = client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=text
        )

        # Extract vector
        embedding = resp.data[0].embedding
        return embedding
    except Exception as e:
        logger.error("Failed to embed text: %s", e)
        raise


def embed_texts(texts: list[str]):
    """
    Embed a batch of texts using OpenAI's embedding model.
    Returns a list of vectors (list of list of floats).
    """
    try:
        # Call OpenAI embeddings API once for the entire batch
        resp = client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=texts
        )

        # Extract embeddings for all inputs
        embeddings = [item.embedding for item in resp.data]
        return embeddings
    except Exception as e:
        logger.error("Failed to embed texts: %s", e)
        raise
