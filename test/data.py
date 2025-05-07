import requests
from bs4 import BeautifulSoup
from langchain.chains import RetrievalQA


# Step 1: Scrape data from the URL
url = "https://www.scrapethissite.com/pages/simple/"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Extract data (e.g., country names and details)
countries = []
for country_div in soup.find_all('div', class_='country'):
    # Extract country details
    countries.append({
        "name": country_div.find('h3', class_='country-name').text.strip(),
        "capital": country_div.find('span', class_='country-capital').text.strip(),
        "population": country_div.find('span', class_='country-population').text.strip(),
        "area": country_div.find('span', class_='country-area').text.strip()
    })

import chromadb
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions

# Initialize ChromaDB client
client = chromadb.Client()

# use directly
google_ef  = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key="AIzaSyAO-5OmSfaSEZwwwuRxpjafmFzdsmWBENY")
google_ef(["document1","document2"])

# pass documents to query for .add and .query
collection = client.create_collection(name="name", embedding_function=google_ef)
collection = client.get_collection(name="name", embedding_function=google_ef)

# Prepare data for insertion
documents = []
metadatas = []
ids = []

for i, country in enumerate(countries):
    # Create a text representation of the country for the document
    country_text = f"Country: {country['name']}, Capital: {country['capital']}, Population: {country['population']}, Area: {country['area']}"
    documents.append(country_text)
    
    # Store the original data as metadata
    metadatas.append({
        "name": country["name"],
        "capital": country["capital"],
        "population": country["population"],
        "area": country["area"]
    })
    
    # Create a unique ID for each entry
    ids.append(f"country_{i}")

# Add all countries to ChromaDB at once
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

from langchain.chat_models import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model_name='gemini-2.0-flash')  # Or your chosen Gemini model
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=collection.as_retriever())

def get_response(query):
    return qa.run(query)

# 5. Build Chatbot Interface
while True:
    query = input("Enter your query: ")
    response = get_response(query)
    print(response)