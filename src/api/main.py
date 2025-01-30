# src/api/main.py
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import numpy as np

from ..database.db import get_db, Paper, UserFeedback
from ..models.encoder import TextEncoder

app = FastAPI(
    title="Research Assistant API",
    description="API for academic paper recommendations and analysis"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class PaperBase(BaseModel):
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    arxiv_id: str

    class Config:
        from_attributes = True

class PaperSearch(BaseModel):
    query: str
    limit: Optional[int] = 10
    min_similarity: Optional[float] = 0.5

class FeedbackCreate(BaseModel):
    paper_id: int
    user_id: str
    rating: float
    # rating: bool  # True if liked, False if disliked
    feedback_text: Optional[str] = None

class SearchResponse(BaseModel):
    paper: PaperBase
    similarity_score: float

# Initialize text encoder 
encoder = TextEncoder()

def get_paper_embedding(paper: Paper) -> np.ndarray:
    """Generate embedding for a paper"""
    return encoder.encode_paper(paper.title, paper.abstract)

def update_paper_embedding(db: Session, paper_id: int):
    """Background task to update paper embedding"""
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if paper:
        embedding = get_paper_embedding(paper)
        paper.embedding = embedding.tolist()
        db.commit()

@app.get("/")
def read_root():
    return {"message": "Welcome to UC Riverside Research Assistant API"}

@app.post("/search/", response_model=List[SearchResponse])
async def search_papers(search: PaperSearch, db: Session = Depends(get_db)):
    """
    Search for papers based on query text
    Returns similar papers using embedding similarity
    """
    try:
        # Generate query embedding
        query_embedding = encoder.encode(search.query)
        print("Query embedding shape:", query_embedding.shape)

        # Get all papers with embeddings
        papers = db.query(Paper).filter(Paper.embedding.isnot(None)).all()
        print(f"Found {len(papers)} papers with embeddings")
        
        if not papers:
            raise HTTPException(status_code=404, detail="No papers found in database")

        # Get paper embeddings and compute similarities
        results = []
        for paper in papers:
            paper_embedding = np.array(paper.embedding)
            print("Paper embedding shape:", paper_embedding.shape)

            similarity = float(encoder.compute_similarity(query_embedding[0], paper_embedding[0]))
            
            if similarity >= search.min_similarity:
                results.append({
                    "paper": paper,
                    "similarity_score": similarity
                })

        # Sort by similarity and limit results
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        results = results[:search.limit]

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/papers/", response_model=PaperBase)
async def create_paper(
    paper: PaperBase,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Create a new paper and generate its embedding in the background
    """
    try:
        db_paper = Paper(**paper.model_dump())
        db.add(db_paper)
        db.commit()
        db.refresh(db_paper)
        
        # Generate embedding in background
        background_tasks.add_task(update_paper_embedding, db, db_paper.id)
        
        return db_paper
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/feedback/")
async def submit_feedback(feedback: FeedbackCreate, db: Session = Depends(get_db)):
    """Submit user feedback for a paper"""
    try:
        db_feedback = UserFeedback(**feedback.model_dump())
        db.add(db_feedback)
        db.commit()
        return {"message": "Feedback submitted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/papers/{paper_id}")
# async def get_paper(paper_id: int, db: Session = Depends(get_db)):
#     """
#     Get details of a specific paper
#     """
#     paper = db.query(Paper).filter(Paper.id == paper_id).first()
#     if paper is None:
#         raise HTTPException(status_code=404, detail="Paper not found")
#     return paper

@app.get("/recommendations/{user_id}", response_model=List[SearchResponse])
async def get_recommendations(
    user_id: str,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get personalized recommendations based on user's feedback history
    """
    try:
        # Get user's highly rated papers
        liked_papers = db.query(Paper).join(UserFeedback)\
            .filter(UserFeedback.user_id == user_id)\
            .filter(UserFeedback.rating >= 4.0)\
            .all()

        if not liked_papers:
            # If no feedback, return recent papers
            return db.query(Paper)\
                .order_by(desc(Paper.published_date))\
                .limit(limit)\
                .all()

        # Generate average embedding from liked papers
        liked_embeddings = [np.array(paper.embedding) for paper in liked_papers if paper.embedding]
        if not liked_embeddings:
            raise HTTPException(status_code=404, detail="No embeddings found for liked papers")
            
        average_embedding = np.mean(liked_embeddings, axis=0)

        # Find similar papers
        papers = db.query(Paper).filter(Paper.embedding.isnot(None)).all()
        results = []
        
        for paper in papers:
            if paper not in liked_papers:  # Exclude already liked papers
                paper_embedding = np.array(paper.embedding)
                similarity = float(encoder.compute_similarity(average_embedding, paper_embedding[np.newaxis, :])[0])
                results.append({
                    "paper": paper,
                    "similarity_score": similarity
                })

        # Sort and limit results
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:limit]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)