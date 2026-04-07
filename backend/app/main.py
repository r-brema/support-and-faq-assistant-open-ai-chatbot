from fastapi import FastAPI
from app.api import chat, ingest
from fastapi.middleware.cors import CORSMiddleware



# Initialize FastAPI app with metadata
app = FastAPI(title="Healthcare FAQ Chatbot")

# Enable CORS (Cross-Origin Resource Sharing) so the frontend (Deep Chat running on localhost:5173) can communicate with this backend API without being blocked by browser policies
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",
                    "https://support-and-faq-assistant-open-ai-c.vercel.app"],  # frontend origin allowed to access this API
    allow_methods=["*"],  # allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # allow all custom headers
)

# # Register API routers
# /chat -> chatbot user query endpoint
app.include_router(chat.router, prefix="/chat", tags=["chat"])

# # /ingest -> document ingestion endpoint (optional, can be disabled in production if not needed)
# app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
# @app.get("/")
# async def root():
#     return {"message": "Hello, FastAPI!"}
# # Simple health check endpoint to confirm API is running


# main.py


@app.get("/ping")
def ping():
    return {"status": "okkk"}
@app.get("/")
async def root():
    return {"message": "Hello, FastAPI!"}


# Entry point for uvicorn
if __name__ == "__main__":
    import uvicorn
    PORT = int(os.environ.get("PORT", 8000))  # Render sets PORT env variable
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT)