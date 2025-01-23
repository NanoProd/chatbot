#utils/load_knowledge_items.py
import pandas as pd
from database.connection import engine
from database.connection import SessionLocal
from database.models import Document
from rag.embeddings import EmbeddingHandler
import json
from database.models import Base

def load_knowledge_items(csv_path: str):
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    
    # Initialize embedding handler
    embedding_handler = EmbeddingHandler()
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    with SessionLocal() as db:
        for _, row in df.iterrows():
            # Generate embeddings for main knowledge item text
            embedding = embedding_handler.get_embedding(row['ki_text'])
            
            # Create document
            doc = Document(
                ki_topic=row['ki_topic'],
                ki_text=row['ki_text'],
                alt_ki_text=row['alt_ki_text'],
                bad_ki_text=row['bad_ki_text'],
                embedding=json.dumps(embedding),
                meta_info=json.dumps({
                    "topic": row['ki_topic']
                })
            )
            
            db.add(doc)
        
        db.commit()

if __name__ == "__main__":
    load_knowledge_items("data/synthetic_knowledge_items.csv")