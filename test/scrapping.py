import streamlit as st
import requests
from bs4 import BeautifulSoup
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import google.generativeai as genai

# Set page configuration
st.set_page_config(
    page_title="Country Information Chatbot",
    page_icon="🌍",
    layout="centered"
)

# API key - Removed hardcoded key. Will use st.secrets instead.
# Make sure you have a .streamlit/secrets.toml file with your GOOGLE_API_KEY

# Initialize session state for chat history if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def initialize_resources():
    """Initialize and cache ChromaDB and Gemini resources"""
    # Retrieve API key from secrets
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except KeyError:
        st.error("GOOGLE_API_KEY not found in st.secrets. Please add it to your .streamlit/secrets.toml file.")
        st.stop() # Stop execution if key is missing

    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash') # Updated model name if needed, check Gemini docs

    # Initialize ChromaDB
    client = chromadb.Client()
    google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=api_key)

    # Try to get the collection, create it if it doesn't exist
    try:
        # Ensure collection name matches where it's used/created
        collection = client.get_collection(name="country_data", embedding_function=google_ef)
    except Exception as e:
        # More specific exception handling might be needed depending on chromadb version
        st.info(f"Collection 'country_data' not found or error accessing it: {e}. It might be created later.")
        collection = None # Indicate collection doesn't exist yet

    return client, google_ef, model, collection

def scrape_and_store_data():
    """Scrape country data and store in ChromaDB"""
    client, google_ef, _, _ = initialize_resources() # Get client and ef

    # Scrape data
    url = "https://www.scrapethissite.com/pages/simple/"
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from {url}: {e}")
        return 0 # Return 0 countries processed

    soup = BeautifulSoup(response.text, 'html.parser')

    countries = []
    for country_div in soup.find_all('div', class_='country'):
        try:
            name = country_div.find('h3', class_='country-name').text.strip()
            capital = country_div.find('span', class_='country-capital').text.strip()
            population = country_div.find('span', class_='country-population').text.strip()
            area = country_div.find('span', class_='country-area').text.strip()
            countries.append({
                "name": name,
                "capital": capital,
                "population": population,
                "area": area
            })
        except AttributeError:
            st.warning("Skipping an entry due to missing data during scraping.")
            continue # Skip this entry if any element is missing

    if not countries:
        st.warning("No country data scraped.")
        return 0

    collection_name = "country_data" # Use a consistent collection name

    # Try to delete existing collection
    try:
        client.delete_collection(name=collection_name)
        st.info(f"Existing collection '{collection_name}' deleted.")
    except Exception as e:
        st.info(f"Could not delete collection '{collection_name}' (it might not exist): {e}")
        pass # Ignore if deletion fails (e.g., collection doesn't exist)

    # Create collection and store data
    try:
        collection = client.create_collection(name=collection_name, embedding_function=google_ef)
        st.info(f"Collection '{collection_name}' created.")
    except Exception as e:
        st.error(f"Failed to create collection '{collection_name}': {e}")
        return 0 # Cannot proceed without collection

    # Prepare data for insertion
    documents = []
    metadatas = []
    ids = []

    for i, country in enumerate(countries):
        # Ensure data types are consistent if needed by ChromaDB or your logic
        country_text = f"Country: {country['name']}, Capital: {country['capital']}, Population: {country['population']}, Area: {country['area']}"
        documents.append(country_text)

        metadatas.append({
            "name": country["name"],
            "capital": country["capital"],
            "population": country["population"], # Consider converting to int/float if needed
            "area": country["area"] # Consider converting to float if needed
        })

        ids.append(f"country_{i}")

    # Add all countries to ChromaDB at once
    try:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        st.success(f"Added {len(countries)} countries to ChromaDB.")
    except Exception as e:
        st.error(f"Failed to add data to ChromaDB: {e}")
        return 0 # Return 0 as data wasn't stored successfully

    # Update the cached collection reference after creation/update
    # This part is tricky with @st.cache_resource. A full rerun might be needed,
    # or a more complex state management approach if immediate reflection is required
    # without rerunning the whole app after scraping.
    # For simplicity, let's rely on the user potentially needing to interact again
    # or the next call to initialize_resources picking up the new collection state.

    return len(countries)


def retrieve_context(query, n_results=3):
    """Retrieve relevant context from ChromaDB"""
    # Get potentially updated collection reference
    client, google_ef, model, collection = initialize_resources()

    if collection is None:
        # Attempt to get the collection again, in case it was created by scrape_and_store_data
        try:
            collection = client.get_collection(name="country_data", embedding_function=google_ef)
        except Exception:
             st.warning("No data found in ChromaDB. Please click 'Refresh Country Data' first.")
             return "No data available."

    # Check if collection exists but is empty
    if collection.count() == 0:
        st.warning("Database is empty. Please click 'Refresh Country Data'.")
        return "No data available."

    # Query ChromaDB for relevant documents
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
    except Exception as e:
        st.error(f"Error querying ChromaDB: {e}")
        return "Error retrieving data."

    # Format the retrieved information into a context string
    if not results or not results.get("documents") or not results["documents"][0]:
        return "Could not find relevant information for your query."

    context = "Here is some information that might help answer the question:\n\n"

    for doc in results["documents"][0]:
        context += f"- {doc}\n" # Use bullet points for clarity

    return context

def rag_chatbot(query):
    """Generate response using RAG pattern with Gemini"""
    # Get model reference
    _, _, model, _ = initialize_resources()

    # Step 1: Retrieve relevant context from vector database
    context = retrieve_context(query)

    # Handle cases where context retrieval failed or returned no data
    if context in ["No data available.", "Error retrieving data.", "Could not find relevant information for your query."]:
        # Optionally provide a direct response or try answering without context
        # For now, let's inform the user based on the context message
        # Or try a direct LLM call without context as a fallback (might hallucinate)
        # return context # Return the error/status message directly

        # Fallback: Try answering without specific context (might be less accurate)
        st.warning(f"Could not retrieve specific context ({context}). Trying to answer generally.")
        prompt = f"""You are a helpful assistant answering questions about countries.
        Answer the following question: {query}
        If you don't know the answer, say so.
        Answer:"""
    else:
        # Step 2: Create a prompt that includes the retrieved context
        prompt = f"""You are a helpful assistant that answers questions about countries using ONLY the provided context.
        If the context doesn't contain the answer, state that the information is not available in the provided data. Do not make up information.

        Context:
        {context}

        Question: {query}

        Answer:"""

    # Step 3: Generate a response using Gemini with the augmented prompt
    try:
        response = model.generate_content(prompt)
        # Accessing the text part might differ slightly based on the genai library version
        # Check the structure of the 'response' object if response.text fails
        return response.text
    except Exception as e:
        st.error(f"Error generating response from Gemini: {e}")
        # Check for specific API errors (e.g., quota, invalid key)
        if "API key not valid" in str(e):
             st.error("The provided Google API Key is invalid. Please check your secrets.toml file.")
        return "Sorry, I encountered an error while generating the response."


# Create the Streamlit UI
st.title("🌍 Country Information Chatbot")
st.subheader("Ask questions about countries around the world!")

# Sidebar with data management
with st.sidebar:
    st.header("Data Management")
    if st.button("Refresh Country Data"):
        with st.spinner("Scraping and storing data... This may take a moment."):
            # Clear cache before scraping to ensure resources are re-initialized if needed
            # Note: This clears ALL cached resources, use with caution if you have others.
            # st.cache_resource.clear() # Uncomment if necessary, but might impact performance
            count = scrape_and_store_data()
            if count > 0:
                st.success(f"Successfully scraped and stored {count} countries!")
                # Force rerun to potentially update collection status display
                st.rerun()
            else:
                st.error("Data scraping and storing failed. Check logs for details.")

    st.markdown("---")
    st.info("Remember to add your GOOGLE_API_KEY to `.streamlit/secrets.toml`")
    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("Ask a question about any country..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        # Use a spinner during generation
        with st.spinner("Thinking..."):
            # Ensure resources are initialized before attempting RAG
            # The call within rag_chatbot handles this via initialize_resources
            response = rag_chatbot(prompt)
            st.markdown(response) # Display the response text

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Initial check and instructions
# Get client and collection status without triggering full resource init if possible
# This check might be simplified or integrated differently depending on exact needs
try:
    # Attempt a lightweight check if possible, e.g., just getting the client
    # Note: initialize_resources is cached, so this won't re-run heavy init unless cleared/first run
    client, _, _, collection_on_load = initialize_resources()
    collection_exists = False
    collection_count = 0
    if collection_on_load:
        collection_exists = True
        collection_count = collection_on_load.count()
    elif client: # If client exists, try getting collection again
         try:
            collection_check = client.get_collection(name="country_data")
            collection_exists = True
            collection_count = collection_check.count()
         except Exception:
            collection_exists = False # Collection likely doesn't exist

    if not collection_exists:
         st.warning("⚠️ Database collection not found. Please click 'Refresh Country Data' in the sidebar.")
    elif collection_count == 0:
         st.warning("⚠️ Database is empty. Please click 'Refresh Country Data' in the sidebar.")

    if not st.session_state.messages:
        st.info("👋 Hello! Ask me anything about countries. Use the sidebar to manage data.")

except Exception as e:
    # Catch potential errors during the initial check (e.g., API key issue on first load)
    # Error related to API key should be handled within initialize_resources now.
    st.error(f"An error occurred during initial setup: {e}")
    st.info("Please ensure your API key is correctly set in secrets and try refreshing.")