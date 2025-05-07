import openai

class ResponseGenerator:
    def __init__(self, api_key):
        openai.api_key = api_key

    def generate_response(self, context, user_query):
        prompt = f"Context: {context}\nUser: {user_query}\nChatbot:"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()