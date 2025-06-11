import os
from typing import List, Optional
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

class PineconeStore:
    def __init__(
        self,
        index_name: str,
        namespace: Optional[str] = None,
        dimension: int = 1024,
        embedding_model: str = "multilingual-e5-large",
        custom_embeddings: Optional[Embeddings] = None,
    ):
        """Initialize the Pinecone vector store.
        
        Args:
            index_name: Name of the Pinecone index
            namespace: Optional namespace for the vectors
            dimension: Dimension of the embeddings
            embedding_model: Pinecone embedding model to use (default: multilingual-e5-large)
            custom_embeddings: Optional custom embedding model (if not using Pinecone's)
        """
        self.index_name = index_name
        self.namespace = namespace
        self.dimension = dimension
        
        # Initialize Pinecone client
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        self.pc = Pinecone(api_key=api_key)
        
        # Use Pinecone's integrated embeddings or custom embeddings if provided
        if custom_embeddings:
            self.embeddings = custom_embeddings
        else:
            self.embeddings = PineconeEmbeddings(
                model=embedding_model,
                dimension=dimension,
            )
        
        # Check if index exists, if not create it
        if self.index_name not in [index.name for index in self.pc.list_indexes()]:
            print(f"Creating new Pinecone index: {self.index_name}")
            self.pc.create_index_for_model(
                        name=index_name,
                        cloud="aws",
                        region="us-east-1",
                        embed={
                            "model":"multilingual-e5-large",
                            "field_map":{"text": "chunk_text"},
                        }
                    )
        
        # Initialize the vector store
        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace=self.namespace,
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        self.vector_store.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        return self.vector_store.similarity_search(query, k=k)
    
    def delete_index(self) -> None:
        """Delete the Pinecone index."""
        self.pc.delete_index(self.index_name)