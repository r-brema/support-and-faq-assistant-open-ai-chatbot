import random, logging, time
import os
from langdetect import detect, LangDetectException
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from app.core.config import settings

logger = logging.getLogger(__name__)

def get_openai_client() -> OpenAI:
    """
    Create and return an OpenAI client using the API key.

    Raises:
        RuntimeError: If no API key is found or client initialization fails.

    Returns:
        OpenAI: Initialized OpenAI client object.
    """
    
    # Fetch API key from environment, fallback to settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", getattr(settings, "OPENAI_API_KEY", ""))
    
    # Validate API key exists
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY not set in environment or settings. Please provide your OpenAI API key to use the service. "
        )
    
    # Initialize the OpenAI client
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        return client
    except Exception as e:
        raise RuntimeError("Failed to initialize OpenAI client") from e


# -----------------------------
# Retry helper for OpenAI API
# -----------------------------
def retryable(func, *args, **kwargs):
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

    for attempt in range(max_attempts):
        try:
            # Attempt to call the API function
            return func(*args, **kwargs)
        except (RateLimitError, APITimeoutError, APIError) as e:
            # Catch common transient errors
            last_err = e
            # Wait for the current delay before retrying
            wait_time = delay + random.uniform(0, 0.5)  # jitter
            if logger:
                logger.warning(f"OpenAI API call failed (attempt {attempt}/{max_attempts}): {e}. Retrying in {wait_time:.2f}s")
                
            time.sleep(wait_time)
            # Increase the delay exponentially for next retry
            delay *= backoff

    # If all retries fail, raise the last exception
    raise last_err



def detect_intent(text: str):
    text_lower = text.lower().strip()

    # Greeting detection
    greetings_en = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    greetings_fr = ["bonjour", "salut", "coucou", "bonsoir"]

    farewells_en = ["bye", "goodbye", "see you", "take care", "see you later"]
    farewells_fr = ["au revoir", "à bientôt", "à plus", "salut"]

    # Check greetings/farewells explicitly
    if text_lower in greetings_en:
        return {"intent": "greeting", "language": "en"}
    if text_lower in greetings_fr:
        return {"intent": "greeting", "language": "fr"}
    if text_lower in farewells_en:
        return {"intent": "farewell", "language": "en"}
    if text_lower in farewells_fr:
        return {"intent": "farewell", "language": "fr"}

    # Fallback to language detection for FAQs
    try:
        # Detect language
        detected_lang = detect(text)
        if detected_lang not in ["en", "fr"]:
            detected_lang = "en"  # default fallback
    except LangDetectException:
        detected_lang = "en"
    
    return {"intent": "faq", "language": detected_lang}
