class Chatbot:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def get_response(self, user_input):
        retrieved_docs = self.retriever.retrieve(user_input)
        response = self.generator.generate(retrieved_docs, user_input)
        return response

    def chat(self):
        print("Welcome to the RAG Chatbot! Type 'exit' to end the conversation.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            response = self.get_response(user_input)
            print(f"Chatbot: {response}")