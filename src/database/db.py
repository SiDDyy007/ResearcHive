# src/database/db.py
from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

# Database URLs
POSTGRES_URL = os.getenv('POSTGRES_URL', 'postgresql://user:password@localhost:5432/research_db')
MONGODB_URL = os.getenv('MONGODB_URL', 'mongodb://localhost:27017/')

# SQLAlchemy setup
engine = create_engine(POSTGRES_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Paper(Base):
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, index=True)
    arxiv_id = Column(String, unique=True, index=True)
    title = Column(String)
    abstract = Column(String)
    authors = Column(JSON)  # Store as JSON array
    categories = Column(JSON)  # Store as JSON array
    published_date = Column(DateTime)
    embedding = Column(JSON)  # Store vector embedding as JSON array
    
class UserFeedback(Base):
    __tablename__ = "user_feedback"

    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(Integer, index=True)
    user_id = Column(String, index=True)
    rating = Column(Float)  # 1-5 rating
    # rating = Column(bool)  # True if liked, False if disliked
    timestamp = Column(DateTime, default=datetime.utcnow)
    feedback_text = Column(String, nullable=True)

def init_db():
    """Initialize the database"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized!")