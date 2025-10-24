import os

from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding, TextEmbedding
from typing import List, Dict
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class SemanticLongTermMemory:
    def __init__(
            self,
            collection_name: str = "long_term_memory",
            qdrant_url: str = "http://localhost:6333",
            sparse_model: str = "prithvida/Splade_PP_en_v1",
            dense_model: str = "BAAI/bge-small-en-v1.5"
    ):
        """
        Initialize LongTermMemory with Qdrant client and embedding models.
        
        Args:
            collection_name: Name of the Qdrant collection
            qdrant_url: URL of the Qdrant instance
            sparse_model: Model for sparse embeddings
            dense_model: Model for dense embeddings
        """
        self.client = QdrantClient(url=qdrant_url, api_key=os.environ.get("QDRANT_API_KEY"))
        self.collection_name = collection_name

        # Initialize embedding models
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model)
        self.dense_model = TextEmbedding(model_name=dense_model)

        # Create collection if it doesn't exist
        self._create_collection()

    def _create_collection(self):
        """Create Qdrant collection with hybrid search configuration."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "text-dense": models.VectorParams(
                        size=384,  # bge-small-en-v1.5 dimension
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "text-sparse": models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    )
                },
            )

    def insert(self, text: str, metadata: Dict = None) -> str:
        """
        Insert a text document into Qdrant.
        
        Args:
            text: The text to insert
            metadata: Optional metadata dictionary
            
        Returns:
            The ID of the inserted document
        """
        # Generate embeddings
        dense_embedding = list(self.dense_model.embed([text]))[0].tolist()
        sparse_embedding = list(self.sparse_model.embed([text]))[0]

        # Create point ID
        import uuid
        point_id = str(uuid.uuid4())

        # Prepare payload
        payload = {"text": text}
        if metadata:
            payload.update(metadata)

        # Insert into Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={
                        "text-dense": dense_embedding,
                        "text-sparse": models.SparseVector(
                            indices=sparse_embedding.indices.tolist(),
                            values=sparse_embedding.values.tolist()
                        )
                    },
                    payload=payload,
                )
            ],
        )

        return point_id

    def retrieve(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Retrieve relevant documents for a given query using hybrid search.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing document text and metadata
        """
        # Generate query embeddings
        dense_query = list(self.dense_model.embed([query]))[0].tolist()
        sparse_query = list(self.sparse_model.embed([query]))[0]

        # Perform hybrid search
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_query,
                    using="text-dense",
                    limit=limit,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_query.indices.tolist(),
                        values=sparse_query.values.tolist()
                    ),
                    using="text-sparse",
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )

        # Format results
        documents = []
        for point in results.points:
            documents.append({
                "id": point.id,
                "text": point.payload.get("text"),
                "score": point.score,
                "metadata": {k: v for k, v in point.payload.items() if k != "text"}
            })

        return documents


# Example usage
# if __name__ == "__main__":
#     # Initialize memory
#     memory = LongTermMemory()
#
#     # Insert some documents
#     doc_id1 = memory.insert(
#         "Python is a high-level programming language.",
#         metadata={"category": "programming"}
#     )
#     doc_id2 = memory.insert(
#         "Machine learning is a subset of artificial intelligence.",
#         metadata={"category": "AI"}
#     )
#
#     print(f"Inserted documents: {doc_id1}, {doc_id2}")
#
#     # Retrieve relevant documents
#     results = memory.retrieve("What is Python?", limit=3)
#
#     print("\nSearch Results:")
#     for result in results:
#         print(f"Score: {result['score']:.4f}")
#         print(f"Text: {result['text']}")
#         print(f"Metadata: {result['metadata']}")
#         print("-" * 50)