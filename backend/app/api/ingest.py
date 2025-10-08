from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.ingestion.json_ingest import run_ingestion
import logging


router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------
# Response Models
# ---------------------------
class IngestResponse(BaseModel):
    status: str
    ingested_count: int
    message: str

# ---------------------------
# Endpoints
# ---------------------------
@router.post("/",
    response_model=IngestResponse,
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
    - `ingested_count`: Total number of unique FAQ items ingested
    - `message`: Description of the result
    """
    try:
        count =  run_ingestion()
        return {"status": "success", "ingested_count": count, "message": "FAQs ingested into ChromaDB"}
    except Exception as e:
        # Log the exception if you have logging configured
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))