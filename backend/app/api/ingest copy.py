from fastapi import APIRouter
from app.ingestion.json_ingest import run_ingestion

router = APIRouter()

@router.post("/",
    summary="Ingest FAQs into ChromaDB",
    description=(
        "Triggers the ingestion process to read FAQ data from the JSON source "
        "and store it into the Chroma vector database for RAG queries."
    ),
    response_description="A confirmation that the ingestion was successful",
)
def ingest_faqs():
    """
    Ingest FAQs Endpoint

    This endpoint allows the backend to populate the ChromaDB with FAQ data.
    It reads the JSON file containing FAQs and inserts them as vector embeddings
    into the database. Intended to be used by admins or during setup.

    **Returns:**
    - `status`: "success" if ingestion completes, "error" if it fails
    - `message`: Description of the result
    """
    try:
        run_ingestion()
        return {"status": "success", "message": "FAQs ingested into ChromaDB"}
    except Exception as e:
        # Log the exception if you have logging configured
        return {"status": "error", "message": f"Ingestion failed: {str(e)}"}