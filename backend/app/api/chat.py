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
        
    
        # Pass question to the RAG pipeline (handles embeddings, vector search, and LLM response)
        # Step 1: Extract messages
        messages = request.messages

        # Step 2: Filter user messages
        user_messages = [m for m in request.messages if m.role == "user"]

        # Step 3: Validate non-empty
        if not user_messages:
            raise ValueError("No user messages provided")

        latest_user_question = user_messages[-1].text.strip()  # pick the most recent
        if not latest_user_question:
            raise ValueError("User message text is empty")

        print("User message text:", latest_user_question)
        result = await query_faq(latest_user_question)
        
        print('rag answer: ', result)
        # Validate RAG response structure
        if not result or "answer" not in result:
            raise HTTPException(status_code=500, detail="Query service returned no response")

        # Log Q/A pair for observability
        logger.info(
            f"Q: {latest_user_question} | "
            f"Lang: {result['language']} | "
            f"A: {result['answer'][:60]}..."  # Log only first 60 chars of answer
        )

        # Return to deep chat accepted format
        return {"role": "bot", "text": result["answer"]}

    except Exception as e:
        # Log error with traceback for debugging
        logger.error(f"Error in /chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
