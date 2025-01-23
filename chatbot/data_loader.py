# data_loader.py
import json
from pinecone import Pinecone
import os
from tqdm import tqdm
import torch
from medical_processor import MedicalDataProcessor
from dotenv import load_dotenv
from typing import List, Dict

class DataLoader:
    def __init__(self, processor: MedicalDataProcessor):
       self.processor = processor
       self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
       # Move model to GPU
       self.processor.model.to(self.device)

    def process_all_data(self):
        data_dir = "data"
        
        # Process MedMCQA
        #medmcqa_path = os.path.join(data_dir, "medmcqa/data.json")
        #if os.path.exists(medmcqa_path):
        #    processed = self.processor.process_medmcqa(medmcqa_path)
        #    for i in range(0, len(processed), 100):
        #        batch = processed[i:i+100]
        #        self.processor.index_documents(batch)

        # Process MedQA
        medqa_path = os.path.join(data_dir, "medqa/questions/us.jsonl")
        if os.path.exists(medqa_path):
            processed = self.processor.process_medqa(medqa_path)
            for i in range(0, len(processed), 100):
                batch = processed[i:i+100]
                self.processor.index_documents(batch)

        # Process textbooks with batch processing
        #textbook_dir = os.path.join(data_dir, "medqa/textbooks/")
        #if os.path.exists(textbook_dir):
        #    i = 0
        #    for file in tqdm(os.listdir(textbook_dir), desc="Processing Textbooks"):
        #        i+=1
        #        if i >=2:
        #            break
        #        if file.endswith('.txt'):
        #            processed = self.processor.process_textbook(os.path.join(textbook_dir, file))
        #            for i in range(0, len(processed), 100):
        #                batch = processed[i:i+100]
        #                self.processor.index_documents(batch)

def main():
    load_dotenv()
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    
    # Create index if doesn't exist
    if 'medical-index' not in pc.list_indexes().names():
        pc.create_index(
            name='medical-index',
            dimension=768,
            metric='cosine'
        )
    
    index = pc.Index("medical-index")
    index.delete(delete_all=True)
    processor = MedicalDataProcessor(index)
    loader = DataLoader(processor)
    loader.process_all_data()

if __name__ == "__main__":
   main()