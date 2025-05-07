import time
from chatbot import Chatbot

def main():
    # Initialize the chatbot
    chatbot = Chatbot()

    print("Welcome to the RAG Chatbot! Type 'exit' to end the conversation.")
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Get response from the chatbot
        response = chatbot.get_response(user_input)
        
        # Print the chatbot's response
        print(f"Chatbot: {response}")
        time.sleep(1)  # Optional: add a delay for a more natural conversation flow

if __name__ == "__main__":
    main()