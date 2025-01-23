from uuid import uuid4
from sentence_transformers import CrossEncoder
import torch
from transformers import AutoTokenizer, AutoModel
import pinecone
import pandas as pd
import json
from typing import Dict, List
import numpy as np

from settings import settings

class MedicalDataProcessor:
    def __init__(self, pinecone_index):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        print("loading tokenizer..")
        self.tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)

        print("loading model..")
        self.model = AutoModel.from_pretrained(settings.embedding_model)
        self.index = pinecone_index
    
    def process_medmcqa(self, file_path: str) -> List[Dict]:
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f]
            
        processed = []
        for item in data:
            correct_option = f"op{chr(96 + item['cop'])}"
            context = f"Question: {item['question']}\nExplanation: {item['exp']}"
            
            # Handle potential missing values
            metadata = {
                'subject': item.get('subject_name', ''),
                'topic': item.get('topic_name', ''),
                'source': 'medmcqa',
                'answer': item.get(correct_option, ''),
                'question_id': item.get('id', ''),
                'text': context
            }
            
            # Remove any null values
            metadata = {k: v for k, v in metadata.items() if v is not None}
            
            processed.append({
                'text': context,
                'metadata': metadata
            })
        return processed

    def process_medqa(self, file_path: str) -> List[Dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f][:5000]
            
        processed = []
        for item in data:
            # Create comprehensive context
            question = item['question']
            
            processed.append({
                'text': question,
                'metadata': {
                    'source': 'medqa',
                    'meta_info': item['meta_info'],
                    'options': ', '.join([f"{k}: {v}" for k, v in item['options'].items()]),
                    'answer': item['answer'],
                }
            })
        return processed

    def process_textbook(self, file_path: str) -> List[Dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Split into manageable chunks (e.g., paragraphs)
        chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
        
        processed = []
        for i, chunk in enumerate(chunks):
            processed.append({
                'text': chunk,
                'metadata': {
                    'source': 'textbook',
                    'chunk_id': i,
                    'file_path': file_path
                }
            })
        return processed

    def generate_embeddings(self, text: str) -> np.ndarray:
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            if hasattr(self, 'device'):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    def index_documents(self, documents: List[Dict]):
       batch_size = 100
       for i in range(0, len(documents), batch_size):
           batch = documents[i:i + batch_size]
           vectors = []
           
           for doc in batch:
               embedding = self.generate_embeddings(doc['text'])
               vectors.append({
                   'id': str(uuid4()),
                   'values': embedding.tolist()[0],
                   'metadata': {**doc['metadata'], 'text': doc['text']}
               })
           
           self.index.upsert(vectors=vectors)

    def search_without_reranking(self, query: str, top_k: int = 3) -> List[Dict]:
        query_embedding = self.generate_embeddings(query)
        results = self.index.query(
            vector=query_embedding.tolist()[0],
            top_k=top_k,
            include_metadata=True
        )

        return [
            {
                "metadata": match["metadata"],
                "score": normalize_score(float(match["score"]))
            }
            for match in results["matches"]
        ]

    def search_with_reranking(self, query: str, top_k: int = 3) -> List[Dict]:
        # Get more initial results for re-ranking
        initial_results = self.search_without_reranking(query, top_k * 2)
        
        # Re-rank using cross-encoder
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        pairs = [[query, result['metadata']['text']] for result in initial_results]
        scores = reranker.predict(pairs)
        
        reranked_results = [
            {
                'metadata': result['metadata'],
                'original_score': float(result['score']),
                'rerank_score': normalize_score(float(score))
            }
            for result, score in zip(initial_results, scores)
        ]
        
        reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
        return reranked_results[:top_k]
    
def normalize_score(score: float) -> float:
    """Normalize a CrossEncoder score to a [0, 1] range."""
    # Define min/max scores based on empirical observations or the model's range
    MIN_SCORE = -15  # Minimum possible score from CrossEncoder
    MAX_SCORE = 0    # Maximum possible score (high similarity)
    normalized = (score - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)
    return max(0.0, min(1.0, normalized))  # Ensure within [0, 1]
