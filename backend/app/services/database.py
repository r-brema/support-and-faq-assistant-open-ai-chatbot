import logging
import os
import chromadb
from app.core.config import settings

logger = logging.getLogger(__name__)

# -------------------------------
# Persistent Chroma client
# -------------------------------
def get_chroma_client() -> chromadb.PersistentClient:
    """
    Initialize and return a Chroma PersistentClient using the configured path.
    
    Raises:
        RuntimeError: If CHROMA_DB_PATH is not set or client initialization fails.
    """
    db_path = getattr(settings, "CHROMA_DB_PATH", None)
    if not db_path:
        raise RuntimeError("CHROMA_DB_PATH is not configured in settings.")

    try:
        client = chromadb.PersistentClient(path=db_path)
        logger.info(f"ChromaDB client initialized with path: {db_path}")
        return client
    except Exception as e:
        logger.exception(f"Failed to initialize ChromaDB client at {db_path}")
        raise RuntimeError(f"ChromaDB initialization failed: {e}") from e
    
def get_chroma_collection(client: chromadb.PersistentClient, name: str = None):
    """
    Get or create a Chroma collection for manual embeddings.
    Prevents Chroma from overriding dimensions.
    """
    # used `get_or_create_collection` to avoid creating a new collection every time
    return client.get_or_create_collection(name=name or settings.CHROMA_COLLECTION,  embedding_function=None)

# def _chunk(lst, size):
#     """Split a list into chunks of given size."""
#     for i in range(0, len(lst), size):
#         yield lst[i:i+size]
    
# def test_chroma_collection(collection_name: str, limit: int = 5):
#     """
#     Utility function to verify data inside a ChromaDB collection.
    
#     Args:
#         persistent_path (str): Path where Chroma persistent DB is stored.
#         collection_name (str): Name of the collection to test.
#         limit (int): Number of sample documents to peek. Default = 5.
#     """
#     print(f"test chhroma Collection functions");
#     try:
#         # Fetch the collection
#         coll = _client.get_or_create_collection(collection_name)
        
#         # List all collections
#         collections = _client.list_collections()
#         print("Collections in ChromaDB:", collections)
#         # Count items
#         try:
#             total = coll.count()
#             print(f"Collection '{collection_name}' contains {total} items.")
#         except Exception as e:
#             print(f" Error during count(): {e}")
#             return

#         if total > 0:
#             try:
#                 samples = coll.peek(limit=limit)
#                 print(f"\nShowing {min(limit, total)} sample(s):")
#                 if samples and "documents" in samples:
#                     for i, doc in enumerate(samples.get("documents", []), 1):
#                         meta = samples.get("metadatas", [])[i-1] if "metadatas" in samples else {}
#                         print(f"\n--- Sample {i} ---")
#                         print(f"Doc   : {doc[:150]}...")
#                         print(f"Meta  : {meta}")
#                         print(f"ID    : {samples['ids'][i-1]}")
#             except Exception as e:
#                 print(f"Error during peek(): {e}")
#         else:
#             print("⚠️ No documents found in this collection.")
#     except Exception as e:
#         print("DEBUG: inside exception  block")
#         print(f"Error testing Chroma collection: {e}")