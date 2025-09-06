from fastapi import APIRouter
from app.ingestion.json_ingest import run_ingestion

router = APIRouter()

@router.post("/")
def ingest_faqs():
    run_ingestion()
    return {"status": "success", "message": "FAQs ingested into ChromaDB"}
