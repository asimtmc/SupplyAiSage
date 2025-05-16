from utils.db_config import Base, engine, init_db
from utils.database import UploadedFile, ForecastResult, ModelParameterCache, SecondarySales

def initialize_database():
    """Initialize the database and create all tables."""
    print("Initializing database...")
    init_db()
    print("Database initialized.")

if __name__ == "__main__":
    initialize_database()