#settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    database_url: str
    embedding_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    pinecone_api_key: str
    pinecone_env: str
    
    class Config:
        env_file = ".env"

settings = Settings()