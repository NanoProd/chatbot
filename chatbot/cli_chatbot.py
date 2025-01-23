# cli_chatbot.py
import requests

BASE_URL = "http://127.0.0.1:8000"

def main():
    print("Medical QA Chatbot. Type 'exit' to quit.")
    use_reranking = input("Enable reranking? (y/n): ").lower() == 'y'
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break

        try:
            response = requests.post(
                f"{BASE_URL}/chat",
                json={
                    "user_input": user_input,
                    "use_reranking": use_reranking
                },
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            print("\nAssistant:", data["response"])
            
            if data["sources"]:
                print("\nSources:")
                for source in data["sources"]:
                    print(f"- {source}")
                    
        except requests.exceptions.ConnectionError:
            print("\nError: Unable to connect to the server. Is it running?")
        except requests.exceptions.Timeout:
            print("\nError: The request timed out. Please try again.")
        except requests.exceptions.RequestException as e:
            print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
   main()