#database/connection.py
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from settings import settings
import logging
from database.models import Base  # Import Base here

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_db_engine():
    try:
        engine = create_engine(
            settings.database_url,
            echo=True,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20
        )
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        # Test the connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("Database connection successful")
        return engine
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise

engine = create_db_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()