# Chatbot Project

This project is a chatbot application built with FastAPI and OpenAI's API. It includes database integration using SQLAlchemy and robust retry mechanisms with Tenacity.

## Prerequisites

Before starting, ensure you have the following installed on your system:

- Python 3.8 or newer
- Git

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/NanoProd/chatbot.git
   cd chatbot

2. **Create a virtual environment**
    ```bash
    python -m venv venv

3. **Activate the virtual environment**
    Windows
    ```bash
    venv\Scripts\activate

    macOS/Linux:
    ```bash
    source venv/bin/activate

4. **Install dependencies**
    ```bash
    pip install -r requirements.txt

5. **Set up environment variables**
    Create a .env file in the root directory:
    ```bash
    touch .env

    Open the .env file and add the following variables:
    env
    OPENAI_API_KEY=your_openai_api_key
    DATABASE_URL=your_database_url
    Replace your_openai_api_key and your_database_url with actual values.

6. **Run the application**
    ```bash
    uvicorn api.main:app --reload
    The application will start at http://127.0.0.1:8000.

7. **Access the API documentation Open your browser and navigate to:**
    ```arduino
    http://127.0.0.1:8000/docs

8. **Additional Commands**
Deactivate the virtual environment
```bash
deactivate