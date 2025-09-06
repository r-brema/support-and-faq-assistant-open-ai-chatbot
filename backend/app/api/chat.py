# app/routers/chat.py
"""
Chat API Router for the RAG system.

This module defines the `/chat` endpoint which accepts a user question
and returns an AI-generated response using the Retrieval-Augmented Generation (RAG) pipeline.

Workflow:
1. Accept a POST request with a user question.
2. Pass the question to the RAG pipeline (`query_faq`).
3. Return the answer in frontend-friendly format: {"role": "bot", "text": "..."}.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.rag import query_faq
import logging

# Create a router instance to attach endpoints to the main app
router = APIRouter()

# Configure logging for this module
logger = logging.getLogger(__name__)

# ---------------------------
# Request & Response Models
# ---------------------------

class ChatRequest(BaseModel):
    """
    Defines the schema for incoming chat requests.

    Attributes:
        question (str): The user’s input question to be processed by the RAG system.
    """
    question: str


class ChatResponse(BaseModel):
    """
    Defines the schema for outgoing chat responses.

    Attributes:
        answer (str): The AI-generated answer from the RAG system.
        language (str): The detected language of the user’s question.
    """
    answer: str
    language: str


# ---------------------------
# Endpoints
# ---------------------------

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handles chat requests using the Retrieval-Augmented Generation (RAG) system.

    Steps:
        1. Receives a question from the client.
        2. Passes the question to the RAG pipeline (`query_faq`).
        3. Logs the request and response for debugging/monitoring.
        4. Returns a structured JSON response in the format:
            {
                "role": "bot",
                "text": "<AI-generated answer>"
            }

    Args:
        request (ChatRequest): The incoming request containing a `question`.

    Returns:
        dict: JSON response with "role" and "text" keys for frontend compatibility.

    Raises:
        HTTPException(500): If the RAG pipeline fails or returns no answer.
    """
    try:
        # Pass question to the RAG pipeline (handles embeddings, vector search, and LLM response)
        result = await query_faq(request.question)

        # Validate RAG response structure
        if not result or "answer" not in result:
            raise HTTPException(status_code=500, detail="Query service returned no response")

        # Log Q/A pair for observability
        logger.info(
            f"Q: {request.question} | "
            f"Lang: {result.get('language')} | "
            f"A: {result['answer'][:60]}..."  # Log only first 60 chars of answer
        )

        # Return to deep chat accepted format
        return {"role": "bot", "text": result["answer"]}

    except Exception as e:
        # Log error with traceback for debugging
        logger.error(f"Error in /chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
