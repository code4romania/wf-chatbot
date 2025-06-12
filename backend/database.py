# database.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, DeclarativeBase, relationship
from datetime import datetime
import uuid

DATABASE_URL = "sqlite:///./api_data.db" # SQLite database file

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class UserQuery(Base):
    __tablename__ = "user_queries"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, default=lambda: str(uuid.uuid4()))
    query_text = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.now(UTC))
    # Store returned answer IDs as a comma-separated string for simplicity
    returned_answer_ids = Column(Text, nullable=True)
    reviews = relationship("UserReview", back_populates="query") # Relation to reviews

class UserReview(Base):
    __tablename__ = "user_reviews"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, default=lambda: str(uuid.uuid4())) # Can link to UserQuery session_id
    answer_id = Column(Integer, nullable=False)
    review_code = Column(Integer, nullable=False) # 1: good, 2: okay, 3-5: worst
    review_text = Column(Text, nullable=True) # Optional text review
    position_in_results = Column(Integer, nullable=True) # To store the rank (1st, 2nd, 3rd...) of the answer
    timestamp = Column(DateTime, default=datetime.now(UTC))

    query_id = Column(Integer, ForeignKey("user_queries.id"), nullable=False)
    query = relationship("UserQuery", back_populates="reviews") # Relationship back to query

Base.registry.configure()

# Function to create tables
def create_db_and_tables():
    Base.metadata.create_all(bind=engine)
    print("Database tables created or already exist.")
