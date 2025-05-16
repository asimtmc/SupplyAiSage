import os
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Database configuration that works both locally and on platforms like Heroku
def get_database_url():
    """
    Get the database URL from environment variables if available,
    otherwise use a local SQLite database.
    """
    # Try to get DATABASE_URL from environment (set by Heroku or other platforms)
    db_url = os.environ.get('DATABASE_URL')
    
    # Fix for Heroku PostgreSQL URL which starts with postgres:// instead of postgresql://
    if db_url and db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql://', 1)
    
    # If no external database URL is found, use local SQLite
    if not db_url:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        db_url = "sqlite:///data/supply_chain.db"
        print(f"Using SQLite database: {db_url}")
    else:
        print(f"Using external database")
    
    return db_url

# Set up database connection
DATABASE_URL = get_database_url()
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith('sqlite') else {},
    pool_pre_ping=True
)

Base = declarative_base()

# Create a session factory
SessionFactory = sessionmaker(bind=engine)

def get_db_session():
    """Get a new database session."""
    return SessionFactory()

def init_db():
    """Create all tables defined in models."""
    Base.metadata.create_all(engine)

# Export for compatibility with existing code
__all__ = ['Base', 'engine', 'SessionFactory', 'get_db_session', 'init_db']