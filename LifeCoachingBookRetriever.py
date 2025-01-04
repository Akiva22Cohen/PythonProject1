###########################################################
# PART 4: Parsing Book & Vector DB (Chroma) Integration
###########################################################

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.api.models import Collection


class LifeCoachingBookRetriever:
    """
    Example class for knowledge retrieval using Chroma.
    """

    def __init__(self, book_file_path: str):
        """
        Initialize the Chroma vector store with the embedded chunks of the book.
        """
        # Initialize Chroma client
        self.client = chromadb.Client(Settings(persist_directory="./chroma_db"))
        self.collection_name = "life_coaching_book"

        # Check if collection exists; otherwise, create it
        if self.collection_name not in self.client.list_collections():
            self.collection = self.client.create_collection(self.collection_name)
        else:
            self.collection = self.client.get_collection(self.collection_name)

        # Read and embed the book content
        self.embed_and_store(book_file_path)

    def embed_and_store(self, book_file_path: str):
        """
        Read the book file, split it into chunks, embed them, and store in Chroma DB.
        """
        embedding_function = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"))

        with open(book_file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Split content into chunks (e.g., 500-word chunks)
        chunks = [content[i:i + 500] for i in range(0, len(content), 500)]

        # Embed chunks and store in Chroma DB
        for idx, chunk in enumerate(chunks):
            self.collection.add(
                documents=[chunk],
                metadatas=[{"chunk_index": idx}],
                ids=[str(idx)],
                embeddings=embedding_function.embed([chunk])
            )

    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> list[str]:
        """
        Given a user query, return the top_k most relevant text chunks from the book.
        """
        embedding_function = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"))
        query_embedding = embedding_function.embed([query])

        results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)
        return results['documents']
