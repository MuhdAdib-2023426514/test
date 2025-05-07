import chromadb

class Retriever:
    def __init__(self, collection):
        self.collection = collection

    def retrieve(self, query, n_results=5):
        # Query the vector database to find relevant documents
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results['documents'], results['metadatas']