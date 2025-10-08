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
from typing import List
from app.services.rag import query_faq
import logging

# Create a router instance to attach endpoints to the main app
router = APIRouter()

# Configure logging for this module
logger = logging.getLogger(__name__)

# ---------------------------
# Request & Response Models
# ---------------------------
class Message(BaseModel):
    role: str
    text: str
    
class ChatRequest(BaseModel):
    """
    Defines the schema for incoming chat requests.

    Attributes:
        question (str): The user’s input question to be processed by the RAG system.
    """
    # question: str
    messages: List[Message]  # DeepChat sends an array of messages


class ChatResponse(BaseModel):
    """
    Defines the schema for outgoing chat responses.

    Attributes:
        answer (str): The AI-generated answer from the RAG system.
        language (str): The detected language of the user’s question.
    """
    role: str
    text: str
    


# ---------------------------
# Endpoints
# ---------------------------
@router.post(
    "/",
    response_model=ChatResponse,
    summary="Chat endpoint for medical FAQ RAG system",
    description="Receives user messages, queries FAQ database, returns AI-generated answer in kind and caring tone."
)
async def chat(request: ChatRequest):
    """
    Handles chat requests using the Retrieval-Augmented Generation (RAG) system.

    Steps:
        1. Receives messages from the client.
        2. Extracts the latest user message.
        3. Validates input and raises HTTP 400 if invalid.
        4. Passes the question to the RAG pipeline (`query_faq`).
        5. Logs the request and response for observability.
        6. Returns a structured JSON response for the frontend.

    Args:
        request (ChatRequest): The incoming request containing an array of messages.

    Returns:
        dict: JSON response with "role" and "text" keys for frontend compatibility.

    Raises:
        HTTPException(400): If no user message is provided or text is empty.
        HTTPException(500): If the RAG pipeline fails or returns no answer.
    """
    try:
        # Pass question to the RAG pipeline (handles embeddings, vector search, and LLM response)
        # Extract messages
        messages = request.messages

        # Filter user messages
        user_messages = [m for m in request.messages if m.role == "user"]

        # Validate non-empty
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user messages provided")

        latest_user_question = user_messages[-1].text.strip()  # pick the most recent
        if not latest_user_question:
            raise HTTPException(status_code=400, detail="User message text is empty")

        logger.info(f"Received user message: {latest_user_question}")
        # Query the RAG system
        result = await query_faq(latest_user_question)
        
        # Validate RAG response structure
        if not result or "answer" not in result:
            raise HTTPException(status_code=500, detail="Query service returned no response")

        # Log Q/A pair for observability
        logger.info(
            f"Q: {latest_user_question} | "
            f"Lang: {result['language']} | "
            f"A: {result['answer'][:60]}..."  # Log only first 60 chars of answer
        )

        # Return in DeepChat-compatible format
        return {"role": "bot", "text": result["answer"]}
    except HTTPException:
        # Re-raise HTTPExceptions (400/500) to prevents the generic exception handler from overwriting our intentional HTTPException response codes.
        raise
    except Exception as e:
        # Log error with traceback for debugging
        logger.error(f"Error in /chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
