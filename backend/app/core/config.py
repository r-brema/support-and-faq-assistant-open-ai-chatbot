import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    db_path = os.path.abspath(os.getenv("CHROMA_DB_PATH"));
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    # Data
    JSON_FAQ_PATH: str = os.getenv("JSON_FAQ_PATH", "app/data/faqs_bilingual.json")

    # Vector DB
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # backend/app
    CHROMA_DB_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../chroma_db"))
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "faq_collections_small")
    CHROMA_METRIC = os.getenv("CHROMA_METRIC", "faq_collections_small")

    # Embeddings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "128"))
    OPENAI_TIMEOUT: float = float(os.getenv("OPENAI_TIMEOUT", "60"))
    
    #RAG Search
    RAG_SIMILARITY_THRESHOLD =  float(os.getenv("RAG_SIMILARITY_THRESHOLD", 1))
    RAG_TOP_K = 3
    RAG_QUERY_TIMEOUT_SECS = 8
    
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()

# print("************ base dir: %s",settings.BASE_DIR)
# print("************ CHROMA_DB_PATH: %s", settings.CHROMA_DB_PATH)
# print("************CHROMA_DB_PATH:", os.path.abspath(settings.CHROMA_DB_PATH))
if not settings.OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set")
