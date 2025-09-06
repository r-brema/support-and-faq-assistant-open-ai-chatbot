import time
from typing import List
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from app.core.config import settings


# Initialize OpenAI client using your API key
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# -----------------------------
# Retry helper for OpenAI API
# -----------------------------
def _retryable(func, *args, **kwargs):
    """
    Wraps an OpenAI API call with retries using exponential backoff.
    
    Args:
        func: The function to call (e.g., client.embeddings.create)
        *args, **kwargs: Arguments to pass to the function
    
    Returns:
        The result of the API call if successful.
    
    Raises:
        The last exception if all retries fail.
    """
    max_attempts = 5        # Maximum number of attempts
    backoff = 1.5           # Multiplier for exponential backoff
    delay = 1.0             # Initial wait time before retrying
    last_err = None         # Store last exception if all retries fail

    for _ in range(max_attempts):
        try:
            # Attempt to call the API function
            return func(*args, **kwargs)
        except (RateLimitError, APITimeoutError, APIError) as e:
            # Catch common transient errors
            last_err = e
            # Wait for the current delay before retrying
            time.sleep(delay)
            # Increase the delay exponentially for next retry
            delay *= backoff

    # If all retries fail, raise the last exception
    raise last_err

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

    # Call the OpenAI embeddings API with retry logic
    resp = _retryable(
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
