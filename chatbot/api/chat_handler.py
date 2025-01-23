#api/ChatHandler.py
from typing import List, Optional
from openai import OpenAI
import openai
from database.models import Conversation, Message
from settings import settings
import json
from medical_processor import MedicalDataProcessor
from tenacity import retry, stop_after_attempt, wait_random_exponential
from fastapi import HTTPException

class ChatHandler:
    def __init__(self, db_session, pinecone_index):
        self.db = db_session
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.medical_processor = MedicalDataProcessor(pinecone_index)

    def create_conversation(self) -> Conversation:
        conversation = Conversation()

        try:
            self.db.add(conversation)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Error creating conversation: {str(e)}"
            )

        return conversation
    

    def get_conversation_history(self, conversation_id: int) -> List[dict]:
        try:
            messages = (
                self.db.query(Message)
                .filter(Message.conversation_id == conversation_id)
                .order_by(Message.timestamp)
                .all()
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching conversation history: {str(e)}"
            )
        
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def save_message(self, conversation_id: int, role: str, content: str, tokens: int, model: str = None):
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tokens_used=tokens,
            model=model
        )
        try:
            print(f"Saving message: {message}")  # Add detailed debug logs
            self.db.add(message)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            print(f"Error saving message: {e}")  # Log the error
            raise HTTPException(
                status_code=500,
                detail=f"Error saving message: {str(e)}"
            )
        return message

    @retry(wait=wait_random_exponential(multiplier=1, max=10), stop=stop_after_attempt(3))
    async def generate_response(self, user_input: str, conversation_id: Optional[int] = None, use_reranking: bool = True) -> tuple:
        try:
            if not conversation_id:
                conversation = self.create_conversation()
                conversation_id = conversation.id
                
            if use_reranking:
                relevant_docs = self.medical_processor.search_with_reranking(user_input)
            else:
                relevant_docs = self.medical_processor.search_without_reranking(user_input)

            print(f"Retrieved documents: {json.dumps(relevant_docs, indent=2)}")

            SIMILARITY_THRESHOLD = 0.3

            filtered_docs = [
                doc for doc in relevant_docs
                if doc.get("rerank_score", doc.get("score", 0)) >= SIMILARITY_THRESHOLD
            ]

            if not filtered_docs:
                assistant_response = (
                    "I'm sorry, I couldn't find any relevant information for your query. "
                    "Please try rephrasing your question or providing more details."
                )
                self.save_message(conversation_id, "user", user_input, 0, "fallback")
                self.save_message(conversation_id, "assistant", assistant_response, 0, "fallback")
                return assistant_response, conversation_id, []

                
            context = "\n".join([doc['metadata']['text'] for doc in relevant_docs])
                
            # Build prompt
            system_prompt = (
                "You are a medical assistant. Use the following context to answer the question. "
                "If you use information from the context, cite the source with its similarity score.\n\nContext:\n{context}"
            ).format(context=context)

            # Get conversation history
            messages = self.get_conversation_history(conversation_id)
                
            # Prepare messages for OpenAI
            openai_messages = [
                {"role": "system", "content": system_prompt},
                *messages,
                {"role": "user", "content": user_input}
            ]

            #add fallback models
            models = [
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k",
                "gpt-4-base"
            ]

            for model in models:
                try:
                    print(f"attempting model: {model}")
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=openai_messages,
                        max_tokens=1000
                    )
                    tokens = getattr(response.usage, 'total_tokens', 0)
                    assistant_response = response.choices[0].message.content
                    print(f"Success with model {model}. Tokens used: {tokens}")
                    break
                except openai.RateLimitError:
                    print(f"Rate limit reached for model {model}. Trying next model.")
                    continue
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error: {str(e)}. Model: {model}, Reranking: {use_reranking}")


            formatted_sources = []
            for doc in filtered_docs:
                metadata = doc.get("metadata", {})
                score = float(doc.get("rerank_score", doc.get("score", 0)))
                source_info = f"Source: {metadata.get('source', 'Unknown')}, Similarity: {score:.2f}"
                
                if "question" in metadata:
                    source_info += f"\nQ: {metadata.get('question', 'Unknown')}\nA: {metadata.get('answer', 'Unknown')}"
                elif "text" in metadata:
                    source_info += f"\nContext: {metadata.get('text', '')[:200]}..."
                
                formatted_sources.append(source_info)

            # Save messages
            self.save_message(conversation_id, "user", user_input, 0, model)
            self.save_message(conversation_id, "assistant", assistant_response, tokens, model) 

            return assistant_response, conversation_id, formatted_sources
        except HTTPException as he:
            print(f"HTTPException in generate_response: {he.detail}")
            raise he
        except Exception as e:
            print(f"Unhandled exception in generate_response: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")