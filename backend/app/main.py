from fastapi import FastAPI
from app.api import chat, ingest
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app with metadata
app = FastAPI(title="Healthcare FAQ Chatbot")

# Enable CORS (Cross-Origin Resource Sharing) so the frontend (Deep Chat running on localhost:5173)
# can communicate with this backend API without being blocked by browser policies
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # frontend origin allowed to access this API
    allow_methods=["*"],  # allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # allow all custom headers
)

# Register API routers
# /chat -> chatbot query endpoint
app.include_router(chat.router, prefix="/chat", tags=["chat"])

# /ingest -> document ingestion endpoint (optional, can be disabled in production if not needed)
app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])

# Simple health check endpoint to confirm API is running
@app.get("/ping")
def ping():
    return {"status": "ok"}
