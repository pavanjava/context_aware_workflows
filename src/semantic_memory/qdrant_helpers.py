from dataclasses import dataclass
from typing import Optional, Union, List, Tuple, Dict, Any

from agno.db.schemas import UserMemory
from agno.knowledge.embedder.fastembed import FastEmbedEmbedder
from agno.utils.log import log_info, log_warning, log_error
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range

def generate_collection_name(prefix: str, table_type: str) -> str:
    """Generate a Qdrant collection name with prefix.

    Args:
        prefix (str): Prefix for the collection name.
        table_type (str): Type of table/collection.

    Returns:
        str: Full collection name.
    """
    return f"{prefix}_{table_type}"


def create_payload_indexes(
        qdrant_client: QdrantClient,
        collection_name: str,
        index_fields: List[str],
) -> None:
    """Create payload indexes for faster filtering.

    Args:
        qdrant_client (QdrantClient): Qdrant client instance.
        collection_name (str): Name of the collection.
        index_fields (List[str]): Fields to create indexes for.
    """
    try:
        for field in index_fields:
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema="keyword",  # Can be adjusted based on field type
            )
    except Exception as e:
        log_warning(f"Error creating payload index for {field}: {e}")

def _build_filter_conditions(
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        topics: Optional[List[str]] = None,
) -> Optional[Filter]:
    """Build Qdrant filter conditions from parameters.

    Args:
        user_id (Optional[str]): User ID to filter by.
        agent_id (Optional[str]): Agent ID to filter by.
        team_id (Optional[str]): Team ID to filter by.
        topics (Optional[List[str]]): Topics to filter by.

    Returns:
        Optional[Filter]: Qdrant Filter object or None if no conditions.
    """
    conditions = []

    if user_id is not None:
        conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))

    if agent_id is not None:
        conditions.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))

    if team_id is not None:
        conditions.append(FieldCondition(key="team_id", match=MatchValue(value=team_id)))

    if topics is not None and len(topics) > 0:
        # Match any of the provided topics
        conditions.append(FieldCondition(key="topics", match=MatchAny(any=topics)))

    if not conditions:
        return None

    return Filter(must=conditions)


def _semantic_search(
        qdrant_client: QdrantClient,
        collection_name: str,
        query_vector: List[float],
        filter_conditions: Optional[Filter] = None,
        limit: int = 100,
        offset: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    """Perform semantic search in Qdrant.

    Args:
        collection_name (str): Name of the collection to search.
        query_vector (List[float]): Query embedding vector.
        filter_conditions (Optional[Filter]): Filter conditions to apply.
        limit (int): Maximum number of results.
        offset (int): Offset for pagination.

    Returns:
        Tuple[List[Dict[str, Any]], int]: List of memory payloads and total count.
    """
    try:
        # Perform vector search
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=filter_conditions,
            limit=limit + offset,  # Fetch extra to handle offset
            with_payload=True,
            with_vectors=False,
        )

        # Get total count with same filters
        count_result = qdrant_client.count(
            collection_name=collection_name,
            count_filter=filter_conditions,
            exact=True,
        )
        total_count = count_result.count

        # Apply offset manually (Qdrant search doesn't support offset directly)
        paginated_results = search_results[offset:offset + limit] if offset > 0 else search_results[:limit]

        # Extract payloads
        memories = [point.payload for point in paginated_results]

        log_info(f"Semantic search found {len(memories)} memories (total: {total_count})")
        return memories, total_count

    except Exception as e:
        log_error(f"Error in semantic search: {e}")
        return [], 0


def _scroll_with_filters(
        qdrant_client: QdrantClient,
        collection_name: str,
        filter_conditions: Optional[Filter] = None,
        limit: int = 100,
        offset: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    """Scroll through Qdrant collection with filters.

    Args:
        collection_name (str): Name of the collection to scroll.
        filter_conditions (Optional[Filter]): Filter conditions to apply.
        limit (int): Maximum number of results.
        offset (int): Offset for pagination.

    Returns:
        Tuple[List[Dict[str, Any]], int]: List of memory payloads and total count.
    """
    try:
        # Get total count first
        count_result = qdrant_client.count(
            collection_name=collection_name,
            count_filter=filter_conditions,
            exact=True,
        )
        total_count = count_result.count

        # Scroll through collection
        scroll_result = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=filter_conditions,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        points, _ = scroll_result
        memories = [point.payload for point in points]

        log_info(f"Scroll found {len(memories)} memories (total: {total_count})")
        return memories, total_count

    except Exception as e:
        log_error(f"Error in scroll: {e}")
        return [], 0


def _get_embedding(text: str) -> List[float]:
    """Generate embedding for text.

    This should be implemented based on your embedding model.
    You can use OpenAI embeddings, sentence transformers, etc.

    Args:
        text (str): Text to embed.

    Returns:
        List[float]: Embedding vector.
    """
    # TODO: Implement your embedding logic here
    return FastEmbedEmbedder().get_embedding(text=text)


def _apply_sorting(
        records: List[Dict[str, Any]],
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Apply sorting to records in memory.

    Args:
        records (List[Dict[str, Any]]): Records to sort.
        sort_by (Optional[str]): Field to sort by.
        sort_order (Optional[str]): 'asc' or 'desc'.

    Returns:
        List[Dict[str, Any]]: Sorted records.
    """
    if not sort_by or not records:
        return records

    reverse = sort_order == "desc" if sort_order else False

    try:
        sorted_records = sorted(
            records,
            key=lambda x: x.get(sort_by, ""),
            reverse=reverse
        )
        log_info(f"Sorted {len(records)} records by {sort_by} ({sort_order or 'asc'})")
        return sorted_records
    except Exception as e:
        log_warning(f"Error sorting records: {e}, returning unsorted")
        return records

@dataclass
class XUserMemory(UserMemory):
    text: str = None
