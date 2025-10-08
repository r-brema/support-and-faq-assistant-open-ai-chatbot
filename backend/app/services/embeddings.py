import logging
from typing import List
from app.services.openai_client import get_openai_client, retryable
from app.core.config import settings

logger = logging.getLogger(__name__)

# -----------------------------
# Main embedding function
# -----------------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Convert a list of text strings into embeddings using OpenAI API.
    
    Args:
        texts (List[str]): List of text strings to embed
        Example: ["Hello world", "How are you?"]
    Returns:
        List[List[float]]: List of embeddings (numerical vectors), one per text
        Example: [[0.1, 0.2, ...], [0.5, -0.1, ...]]
    """
    # Early exit if no texts are provided
    if not texts:
        return []

    # Initialize OpenAI client
    client = get_openai_client()
    
    # Call the OpenAI embeddings API with retry logic
    resp = retryable(
        client.embeddings.create,      # API function
        model=settings.EMBEDDING_MODEL,  # Embedding model to use
        input=texts,                   # Texts to embed
        encoding_format="float",
        timeout=settings.OPENAI_TIMEOUT # API timeout
    )

    # Extract embeddings from the API response
    # resp.data is a list of results, one per input text
    # d.embedding contains the vector for each text
    return [d.embedding for d in resp.data]
