import time
from datetime import date
from typing import Optional, List, Union, Dict, Any, Tuple
from uuid import uuid4

from agno.db import BaseDb, SessionType
from agno.db.schemas import CulturalKnowledge
from agno.db.schemas.evals import EvalRunRecord, EvalFilterType, EvalType
from agno.db.schemas.knowledge import KnowledgeRow
from agno.knowledge.embedder.fastembed import FastEmbedEmbedder
from agno.session import Session
from agno.utils.log import log_error, log_info
from agno.utils.string import generate_id

from src.semantic_memory.qdrant_helpers import _build_filter_conditions, _get_embedding, _semantic_search, _scroll_with_filters, \
    _apply_sorting, generate_collection_name, create_payload_indexes, UserMemory

try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.models import Distance, VectorParams, PointStruct
except ImportError:
    raise ImportError("`qdrant-client` not installed. Please install it using `pip install qdrant-client`")


# def _apply_sorting(records, sort_by, sort_order):
#     log_info(f"{len(records)} records were sorted. {records}")
#     pass
#
#
# def _apply_pagination(records, limit, page):
#     log_info(f"{len(records)} records were paginated. {records}")
#     pass


class QdrantDB(BaseDb):
    def __init__(
            self,
            id: Optional[str] = None,
            qdrant_client: Optional[QdrantClient] = None,
            db_url: Optional[str] = None,
            api_key: Optional[str] = None,
            db_prefix: str = "agno",
            model_name="BAAI/bge-small-en",
            vector_size: int = 1536,  # Default embedding size, can be adjusted
            distance: Distance = Distance.COSINE,
            session_table: Optional[str] = None,
            memory_table: Optional[str] = None,
            metrics_table: Optional[str] = None,
            eval_table: Optional[str] = None,
            knowledge_table: Optional[str] = None,
            culture_table: Optional[str] = None,
    ):
        """
        Interface for interacting with a Qdrant vector database.

        The following order is used to determine the database connection:
            1. Use the qdrant_client if provided
            2. Use the db_url with optional api_key
            3. Raise an error if neither is provided

        Args:
            id (Optional[str]): The ID of the database.
            qdrant_client (Optional[QdrantClient]): Qdrant client instance to use. If not provided a new client will be created.
            db_url (Optional[str]): Qdrant connection URL (e.g., "http://localhost:6333" or cloud URL)
            api_key (Optional[str]): API key for Qdrant cloud instances
            db_prefix (str): Prefix for all Qdrant collection names
            vector_size (int): Size of vectors for collections (default: 1536)
            distance (Distance): Distance metric for vector similarity (default: COSINE)
            session_table (Optional[str]): Name of the collection to store sessions
            memory_table (Optional[str]): Name of the collection to store memories
            metrics_table (Optional[str]): Name of the collection to store metrics
            eval_table (Optional[str]): Name of the collection to store evaluation runs
            knowledge_table (Optional[str]): Name of the collection to store knowledge documents
            culture_table (Optional[str]): Name of the collection to store cultural knowledge

        Raises:
            ValueError: If neither qdrant_client nor db_url is provided.
        """
        if id is None:
            base_seed = db_url or str(qdrant_client)
            seed = f"{base_seed}#{db_prefix}"
            id = generate_id(seed)

        super().__init__(
            id=id,
            session_table=session_table,
            memory_table=memory_table,
            metrics_table=metrics_table,
            eval_table=eval_table,
            knowledge_table=knowledge_table,
            culture_table=culture_table,
        )

        self.db_prefix = db_prefix
        self.vector_size = vector_size
        self.distance = distance
        self.model_name = model_name
        self.embedder = FastEmbedEmbedder(id=self.model_name)

        if qdrant_client is not None:
            self.qdrant_client = qdrant_client
        elif db_url is not None:
            self.qdrant_client = QdrantClient(url=db_url, api_key=api_key)
        else:
            raise ValueError("One of qdrant_client or db_url must be provided")

    def _get_collection_name(self, table_type: str) -> str:
        """Get the active collection name for the given table type."""
        if table_type == "sessions":
            return self.session_table_name

        elif table_type == "memories":
            return self.memory_table_name

        elif table_type == "metrics":
            return self.metrics_table_name

        elif table_type == "evals":
            return self.eval_table_name

        elif table_type == "knowledge":
            return self.knowledge_table_name

        elif table_type == "culture":
            return self.culture_table_name

        else:
            raise ValueError(f"Unknown table type: {table_type}")

    def _ensure_collection_exists(self, collection_name: str) -> bool:
        """Ensure a collection exists, create it if it doesn't.

            Args:
                collection_name (str): Name of the collection to ensure exists.

            Returns:
                bool: True if collection exists or was created successfully.
            """
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(col.name == collection_name for col in collections)

            if not collection_exists:
                # Create collection with vector configuration
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=self.qdrant_client.get_embedding_size(self.model_name),
                                                distance=self.distance),
                )
                log_info(f"Created Qdrant collection: {collection_name}")

            return True

        except Exception as e:
            log_error(f"Error ensuring collection exists: {e}")
            return False

    def _store_record(
                self,
                table_type: str,
                record_id: str,
                data: Dict[str, Any],
                vector: Optional[List[float]] = None,
                index_fields: Optional[List[str]] = None,
        ) -> bool:
        """Generic method to store a record in Qdrant.

            Args:
                table_type (str): The type of table to store the record in.
                record_id (str): The ID of the record to store.
                data (Dict[str, Any]): The data to store in the record payload.
                vector (Optional[List[float]]): Optional vector for the point. If None, uses a zero vector.
                index_fields (Optional[List[str]]): The fields to create payload indexes for.

            Returns:
                bool: True if the record was stored successfully, False otherwise.
            """
        try:
            collection_name = generate_collection_name(prefix=self.db_prefix, table_type=table_type)

            # Ensure collection exists
            self._ensure_collection_exists(collection_name)

            # Use provided vector or create a dummy zero vector
            if vector is None:
                log_error(f"Vector field not provided for {table_type}")

            # Create point with payload
            point = PointStruct(
                id=record_id,
                vector=vector,
                payload=data,
            )

            # Upsert point to collection
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=[point],
            )

            # Create payload indexes if specified
            if index_fields:
                create_payload_indexes(
                    qdrant_client=self.qdrant_client,
                    collection_name=collection_name,
                    index_fields=index_fields,
                )

            return True

        except Exception as e:
            log_error(f"Error storing Qdrant record: {e}")
            return False

    def _get_all_records(self, param):
        pass

    def delete_session(self, session_id: str) -> bool:
        pass

    def delete_sessions(self, session_ids: List[str]) -> None:
        pass

    def get_session(self, session_id: str, session_type: SessionType, user_id: Optional[str] = None,
                    deserialize: Optional[bool] = True) -> Optional[Union[Session, Dict[str, Any]]]:
        pass

    def get_sessions(self, session_type: SessionType, user_id: Optional[str] = None, component_id: Optional[str] = None,
                     session_name: Optional[str] = None, start_timestamp: Optional[int] = None,
                     end_timestamp: Optional[int] = None, limit: Optional[int] = None, page: Optional[int] = None,
                     sort_by: Optional[str] = None, sort_order: Optional[str] = None,
                     deserialize: Optional[bool] = True) -> Union[List[Session], Tuple[List[Dict[str, Any]], int]]:
        pass

    def rename_session(self, session_id: str, session_type: SessionType, session_name: str,
                       deserialize: Optional[bool] = True) -> Optional[Union[Session, Dict[str, Any]]]:
        pass

    def upsert_session(self, session: Session, deserialize: Optional[bool] = True) -> Optional[
        Union[Session, Dict[str, Any]]]:
        pass

    def upsert_sessions(self, sessions: List[Session], deserialize: Optional[bool] = True,
                        preserve_updated_at: bool = False) -> List[Union[Session, Dict[str, Any]]]:
        pass

    def clear_memories(self) -> None:
        pass

    def delete_user_memory(self, memory_id: str, user_id: Optional[str] = None) -> None:
        pass

    def delete_user_memories(self, memory_ids: List[str], user_id: Optional[str] = None) -> None:
        pass

    def get_all_memory_topics(self, user_id: Optional[str] = None) -> List[str]:
        pass

    def get_user_memory(self, memory_id: str, deserialize: Optional[bool] = True, user_id: Optional[str] = None) -> \
            Optional[Union[UserMemory, Dict[str, Any]]]:
        pass

    def get_user_memories(
            self,
            user_id: Optional[str] = None,
            agent_id: Optional[str] = None,
            team_id: Optional[str] = None,
            topics: Optional[List[str]] = None,
            search_content: Optional[str] = None,
            limit: Optional[int] = None,
            page: Optional[int] = None,
            sort_by: Optional[str] = None,
            sort_order: Optional[str] = None,
            deserialize: Optional[bool] = True,
    ) -> Union[List[UserMemory], Tuple[List[Dict[str, Any]], int]]:
        """Get all memories from Qdrant as UserMemory objects with semantic search.

        Args:
            user_id (Optional[str]): The ID of the user to filter by.
            agent_id (Optional[str]): The ID of the agent to filter by.
            team_id (Optional[str]): The ID of the team to filter by.
            topics (Optional[List[str]]): The topics to filter by.
            search_content (Optional[str]): The content to search for semantically.
            limit (Optional[int]): The maximum number of memories to return.
            page (Optional[int]): The page number to return.
            sort_by (Optional[str]): The field to sort by.
            sort_order (Optional[str]): The order to sort by.
            deserialize (Optional[bool]): Whether to deserialize the memories.

        Returns:
            Union[List[UserMemory], Tuple[List[Dict[str, Any]], int]]:
                - When deserialize=True: List of UserMemory objects
                - When deserialize=False: Tuple of (memory dictionaries, total count)
        Raises:
            Exception: If any error occurs while reading the memories.
        """
        try:
            collection_name = generate_collection_name(prefix=self.db_prefix, table_type="memories")

            # Build Qdrant filter conditions
            filter_conditions = _build_filter_conditions(
                user_id=user_id,
                agent_id=agent_id,
                team_id=team_id,
                topics=topics,
            )

            # Determine limit and offset for pagination
            page_size = limit or 100
            offset = ((page or 1) - 1) * page_size if page else 0

            # Perform search or scroll based on whether semantic search is needed
            if search_content:
                # Semantic search with embedding
                query_vector = _get_embedding(search_content)
                memories, total_count = _semantic_search(
                    qdrant_client=self.qdrant_client,
                    collection_name=collection_name,
                    query_vector=query_vector,
                    filter_conditions=filter_conditions,
                    limit=page_size,
                    offset=offset,
                )
            else:
                # Regular scroll with filters
                memories, total_count = _scroll_with_filters(
                    qdrant_client=self.qdrant_client,
                    collection_name=collection_name,
                    filter_conditions=filter_conditions,
                    limit=page_size,
                    offset=offset,
                )

            # Apply sorting if needed (Qdrant doesn't support arbitrary field sorting in search)
            if sort_by:
                memories = _apply_sorting(records=memories, sort_by=sort_by, sort_order=sort_order)

            if not deserialize:
                return memories, total_count

            return [UserMemory.from_dict(record) for record in memories]

        except Exception as e:
            log_error(f"Exception reading memories: {e}")
            raise e

    def get_user_memory_stats(self, limit: Optional[int] = None, page: Optional[int] = None,
                              user_id: Optional[str] = None) -> Tuple[List[Dict[str, Any]], int]:
        pass

    def upsert_user_memory(self, memory: UserMemory, deserialize: Optional[bool] = True) -> Optional[
        Union[UserMemory, Dict[str, Any]]]:
        """Upsert a user memory in Qdrant.
        Args:
            :param memory : (UserMemory) The memory to upsert.
            :param deserialize:

        Returns:
            Optional[UserMemory]: The upserted memory data if successful, None otherwise.
        """
        try:
            if memory.memory_id is None:
                memory.memory_id = str(uuid4())

            data = {
                "user_id": memory.user_id,
                "agent_id": memory.agent_id,
                "team_id": memory.team_id,
                "memory_id": memory.memory_id,
                "memory": memory.memory,
                "topics": memory.topics,
                "updated_at": int(time.time()),
            }

            success = self._store_record(
                table_type="memories", record_id=memory.memory_id, data=data, vector=self.embedder.get_embedding(memory.memory),
                index_fields=["user_id", "agent_id", "team_id", "workflow_id"]
            )

            if not success:
                return None

            if not deserialize:
                return data

            return UserMemory.from_dict(data)

        except Exception as e:
            log_error(f"Error upserting user memory: {e}")
            raise e

    def upsert_memories(self, memories: List[UserMemory], deserialize: Optional[bool] = True,
                        preserve_updated_at: bool = False) -> List[Union[UserMemory, Dict[str, Any]]]:
        pass

    def get_metrics(self, starting_date: Optional[date] = None, ending_date: Optional[date] = None) -> Tuple[
        List[Dict[str, Any]], Optional[int]]:
        pass

    def calculate_metrics(self) -> Optional[Any]:
        pass

    def delete_knowledge_content(self, id: str):
        pass

    def get_knowledge_content(self, id: str) -> Optional[KnowledgeRow]:
        pass

    def get_knowledge_contents(self, limit: Optional[int] = None, page: Optional[int] = None,
                               sort_by: Optional[str] = None, sort_order: Optional[str] = None) -> Tuple[
        List[KnowledgeRow], int]:
        pass

    def upsert_knowledge_content(self, knowledge_row: KnowledgeRow):
        pass

    def create_eval_run(self, eval_run: EvalRunRecord) -> Optional[EvalRunRecord]:
        pass

    def delete_eval_runs(self, eval_run_ids: List[str]) -> None:
        pass

    def get_eval_run(self, eval_run_id: str, deserialize: Optional[bool] = True) -> Optional[
        Union[EvalRunRecord, Dict[str, Any]]]:
        pass

    def get_eval_runs(self, limit: Optional[int] = None, page: Optional[int] = None, sort_by: Optional[str] = None,
                      sort_order: Optional[str] = None, agent_id: Optional[str] = None, team_id: Optional[str] = None,
                      workflow_id: Optional[str] = None, model_id: Optional[str] = None,
                      filter_type: Optional[EvalFilterType] = None, eval_type: Optional[List[EvalType]] = None,
                      deserialize: Optional[bool] = True) -> Union[
        List[EvalRunRecord], Tuple[List[Dict[str, Any]], int]]:
        pass

    def rename_eval_run(self, eval_run_id: str, name: str, deserialize: Optional[bool] = True) -> Optional[
        Union[EvalRunRecord, Dict[str, Any]]]:
        pass

    def clear_cultural_knowledge(self) -> None:
        pass

    def delete_cultural_knowledge(self, id: str) -> None:
        pass

    def get_cultural_knowledge(self, id: str) -> Optional[CulturalKnowledge]:
        pass

    def get_all_cultural_knowledge(self, name: Optional[str] = None, limit: Optional[int] = None,
                                   page: Optional[int] = None, sort_by: Optional[str] = None,
                                   sort_order: Optional[str] = None, agent_id: Optional[str] = None,
                                   team_id: Optional[str] = None) -> Optional[List[CulturalKnowledge]]:
        pass

    def upsert_cultural_knowledge(self, cultural_knowledge: CulturalKnowledge) -> Optional[CulturalKnowledge]:
        pass
