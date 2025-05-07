import requests
from bs4 import BeautifulSoup
import chromadb
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions

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

# Initialize ChromaDB client
client = chromadb.Client()

# Use embedding function
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key="YOUR_API_KEY")
collection = client.create_collection(name="countries", embedding_function=google_ef)

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

print(f"Successfully stored {len(countries)} countries in ChromaDB!")

# Example query to verify the data is stored
result = collection.query(
    query_texts=["What is the population of France?"],
    n_results=1
)
print("\nSample query results:")
print(result)