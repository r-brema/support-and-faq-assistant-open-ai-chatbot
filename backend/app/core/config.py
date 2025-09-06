import os
from dotenv import load_dotenv
load_dotenv()
print('db path: ' , os.getenv("CHROMA_DB_PATH"))
class Settings:
    db_path = os.path.abspath(os.getenv("CHROMA_DB_PATH"));
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    # Data
    JSON_FAQ_PATH: str = os.getenv("JSON_FAQ_PATH", "app/data/faqs_bilingual.json")

    # Vector DB
    CHROMA_DB_PATH: str = db_path
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "faq_collections_small")

    # Embeddings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "128"))
    OPENAI_TIMEOUT: float = float(os.getenv("OPENAI_TIMEOUT", "60"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()

print("Using Chroma DB absolute path: %s", settings.db_path)
print("CHROMA_DB_PATH: %s", settings.CHROMA_DB_PATH)

if not settings.OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set")
