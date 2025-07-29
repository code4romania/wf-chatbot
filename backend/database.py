from datetime import UTC, datetime
import os
import uuid

from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship


# Use environment variable for the database path with fallback
# This allows different configurations for Docker and local development
DEFAULT_DB_PATH = os.path.abspath(os.path.join(os.path.pardir, "data", "api_data.db"))
DB_PATH = os.environ.get("DB_PATH", DEFAULT_DB_PATH)

DATABASE_URL = f"sqlite:///{DB_PATH}"

# Ensure the directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

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
    concat_option_active = Column(Boolean, nullable=False, default=True) # New field to store the concat option used
    reviews = relationship("UserReview", back_populates="query")  # Relation to reviews


class UserReview(Base):
    __tablename__ = "user_reviews"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, default=lambda: str(uuid.uuid4()))  # Can link to UserQuery session_id
    answer_id = Column(Integer, nullable=False)
    review_code = Column(Integer, nullable=False)  # 1: good, 2: okay, 3-5: worst
    review_text = Column(Text, nullable=True)  # Optional text review
    position_in_results = Column(Integer, nullable=True)  # To store the rank (1st, 2nd, 3rd...) of the answer
    timestamp = Column(DateTime, default=datetime.now(UTC))

    query_id = Column(Integer, ForeignKey("user_queries.id"), nullable=False)
    query = relationship("UserQuery", back_populates="reviews")  # Relationship back to query




# Function to create tables
def create_db_and_tables():
    Base.metadata.create_all(bind=engine)
    print("Database tables created or already exist.")
