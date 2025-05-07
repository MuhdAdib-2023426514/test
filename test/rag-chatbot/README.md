# RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot that utilizes a vector database (ChromaDB) to provide accurate and contextually relevant responses to user queries. The chatbot is designed to answer questions about countries, including their names, capitals, populations, and areas.

## Project Structure

```
rag-chatbot
├── src
│   ├── main.py          # Entry point for the chatbot application
│   ├── data.py          # Data scraping and storage
│   ├── chatbot.py       # Main chatbot class
│   ├── retriever.py     # Retrieval logic for querying the vector database
│   ├── generator.py     # Response generation logic
│   └── utils.py         # Utility functions
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd rag-chatbot
   ```

2. **Install dependencies:**
   Ensure you have Python installed, then run:
   ```
   pip install -r requirements.txt
   ```

3. **Run the chatbot:**
   Execute the main script to start the chatbot:
   ```
   python src/main.py
   ```

## Usage Guidelines

- Once the chatbot is running, you can interact with it by typing your queries related to countries.
- The chatbot will retrieve relevant information from the vector database and generate responses based on the data stored.

## Overview of Functionality

- **Data Scraping:** The `data.py` file scrapes country data from a specified URL and stores it in ChromaDB for efficient retrieval.
- **Retrieval Logic:** The `retriever.py` file queries the vector database to find the most relevant documents based on user input.
- **Response Generation:** The `generator.py` file formulates coherent responses using the retrieved documents.
- **Main Chatbot Class:** The `chatbot.py` file integrates the retriever and generator to handle user queries seamlessly.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.