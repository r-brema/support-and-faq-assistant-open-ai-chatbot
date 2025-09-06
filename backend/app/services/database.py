import chromadb 

from app.core.config import settings

# -------------------------------
# Persistent Chroma client
# -------------------------------
_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
def get_collection(name: str = None):
    """
    Get or create a Chroma collection for manual embeddings.
    Prevents Chroma from overriding dimensions.
    """
    # used `get_or_create_collection` to avoid creating a new collection every time
    return _client.get_or_create_collection(name=name or settings.CHROMA_COLLECTION,  embedding_function=None)
def _chunk(lst, size):
    """Split a list into chunks of given size."""
    for i in range(0, len(lst), size):
        yield lst[i:i+size]
    
def test_chroma_collection(persistent_path: str, collection_name: str, limit: int = 5):
    """
    Utility function to verify data inside a ChromaDB collection.
    
    Args:
        persistent_path (str): Path where Chroma persistent DB is stored.
        collection_name (str): Name of the collection to test.
        limit (int): Number of sample documents to peek. Default = 5.
    """
    print(f"✅ test chhroma Collection functions");
    try:
        print("DEBUG: inside try  block")
        # Fetch the collection
        client = chromadb.PersistentClient(path=persistent_path)
        coll = client.get_or_create_collection(collection_name)
        print('coll : ', coll)
        # coll = _client.get_or_create_collection(collection_name)
        print(f"DEBUG: Function started persistent_path: '{persistent_path}',  collection_name = '{ collection_name}' ")
        
        # List all collections
        collections = client.list_collections()
        print("Collections in ChromaDB:", collections)
        # Count items
        try:
            total = coll.count()
            print(f"✅ Collection '{collection_name}' contains {total} items.")
        except Exception as e:
            print(f"❌ Error during count(): {e}")
            return

        if total > 0:
            try:
                samples = coll.peek(limit=limit)
                print(f"\n🔎 Showing {min(limit, total)} sample(s):")
                for i, doc in enumerate(samples.get("documents", []), 1):
                    meta = samples.get("metadatas", [])[i-1] if "metadatas" in samples else {}
                    print(f"\n--- Sample {i} ---")
                    print(f"Doc   : {doc[:150]}...")
                    print(f"Meta  : {meta}")
                    print(f"ID    : {samples['ids'][i-1]}")
            except Exception as e:
                print(f"❌ Error during peek(): {e}")
        else:
            print("⚠️ No documents found in this collection.")
    except Exception as e:
        print("DEBUG: inside exception  block")
        print(f"❌ Error testing Chroma collection: {e}")
