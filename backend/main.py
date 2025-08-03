# main.py
from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
import logging

from fastapi.middleware.cors import CORSMiddleware

# Import your PromptMatcher class
from PromptMatcher import PromptMatcher

# Import database components
from database import SessionLocal, create_db_and_tables, UserQuery, UserReview

# --- Global PromptMatcher Instances ---
# We'll initialize these once when the application starts
prompt_matcher: Optional[PromptMatcher] = None

CITY_LIST = [
        "braila", "brasov", "cluj-napoca", "constanta", "galati",
        "iasi", "oradea", "sibiu", "suceava", "timisoara"
]

# --- FastAPI App Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Initializes the PromptMatcher and creates database tables.
    """
    global prompt_matcher
    logging.info("Starting up API...")
    try:
        logging.info(f"Initializing PromptMatcher")
        prompt_matcher = PromptMatcher()
        if prompt_matcher.full_df is None or prompt_matcher.full_dense_vectors is None or prompt_matcher.full_sparse_vectors is None:
            logging.warning("PromptMatcher (concat=True) initialized but no data was loaded. Check data path and files.")
        else:
            logging.info(f"PromptMatcher (concat=True) successfully loaded {len(prompt_matcher.full_df)} data entries.")
    except FileNotFoundError as e:
        logging.error(f"Failed to initialize PromptMatcher: {e}. Please check BASE_DATA_PATH.")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during PromptMatcher initialization: {e}")
        raise

    logging.info("Creating database tables...")
    create_db_and_tables()
    logging.info("Database tables checked/created.")

    yield  # The application runs
    logging.info("Shutting down API...")
    # Clean-up / resource release if needed (e.g., closing database connections explicitly)


app = FastAPI(
    title="Prompt Matcher API",
    description="API for matching user queries to predefined prompts and responses.",
    version="1.0.0",
    lifespan=lifespan,  # Use the lifespan context manager
)

# CORS
origins = [
    "http://localhost:3000",  # Your Nuxt.js frontend development server's address
    # You might add other origins for production deployment later, e.g.:
    # "https://your-production-frontend.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins
    allow_credentials=True,  # Allows cookies to be included in cross-origin requests
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers in the request
)
# END CORS


# --- Dependency to get DB session ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Pydantic Models for Request/Response Bodies ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=1, ge=1, le=10)  # Get between 1 and 10 results
    session_id: Optional[str] = None  # Allow client to provide session ID
    use_concat_matcher: bool = Field(default=True, description="Whether to use the matcher with concatenated Q&A (True) or separate Q&A (False).")


class MatchedResponse(BaseModel):
    matched_prompt: str
    response: str
    instruction: str
    score: float
    question_id: int
    answer_id: int


class QueryResponse(BaseModel):
    session_id: str
    results: List[MatchedResponse]
    query_id: int


class ReviewRequest(BaseModel):
    session_id: str = Field(..., description="Session ID from the query request.")
    answer_id: int = Field(..., description="The ID of the answer being reviewed.")
    review_code: int = Field(..., ge=1, le=5, description="1: good, 2: okay, 3-5: worst.")
    review_text: Optional[str] = None
    position_in_results: Optional[int] = None  # Position of the answer in the returned list (1-indexed)
    query_id: int = Field(..., description="The ID of the query that generated this answer.")


class ReviewResponse(BaseModel):
    message: str
    review_id: int
    session_id: str


# --- API Endpoints ---


@app.get("/")
async def root():
    return {"message": "Welcome to the Prompt Matcher API! Use /query to get started."}


@app.post("/query", response_model=QueryResponse)
async def query_prompts(request: QueryRequest, db: Session = Depends(get_db)):
    # Select the appropriate PromptMatcher instance


    if prompt_matcher.full_df is None or prompt_matcher.full_dense_vectors is None or prompt_matcher.full_sparse_vectors is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Selected PromptMatcher is not initialized or data is not loaded yet. Please try again later.",
        )

    try:
        #TODO Make me async
        results = prompt_matcher.query(user_prompt=request.query, top_k=request.top_k)

        # Ensure results is always a list for consistent processing
        if not isinstance(results, list):
            results = [results]

        # Extract answer IDs for storage
        returned_answer_ids = ",".join([str(r["answer_id"]) for r in results])

        # Store the user query
        user_query_db = UserQuery(
            session_id=request.session_id if request.session_id else None,  # Use provided or let DB generate
            query_text=request.query,
            returned_answer_ids=returned_answer_ids,
            concat_option_active=True, # Store the option used
        )
        db.add(user_query_db)
        db.commit()
        db.refresh(user_query_db)  # Refresh to get the generated session_id if new

        # Return the response
        return QueryResponse(
            session_id=user_query_db.session_id,  # Ensure we return the actual session_id
            results=[MatchedResponse(**r) for r in results],
            query_id=user_query_db.id,
        )
    except ValueError as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logging.error(f"Error during query processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal server error occurred."
        )


@app.post("/review", response_model=ReviewResponse)
async def submit_review(review_data: ReviewRequest, db: Session = Depends(get_db)):
    try:
        user_review_db = UserReview(
            session_id=review_data.session_id,
            answer_id=review_data.answer_id,
            review_code=review_data.review_code,
            review_text=review_data.review_text,
            position_in_results=review_data.position_in_results,
            query_id=review_data.query_id,
        )
        db.add(user_review_db)
        db.commit()
        db.refresh(user_review_db)

        return ReviewResponse(
            message="Review submitted successfully!", review_id=user_review_db.id, session_id=user_review_db.session_id
        )
    except Exception as e:
        logging.error(f"Error submitting review: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred while submitting review.",
        )
