from agno.db.redis import RedisDb
from src.semantic_memory.qdrant_db import QdrantDB


class ShortTermMemory:
    def __init__(self, time_to_live: int = 60):
        # Setup Redis
        # Initialize Redis db (use the right db_url for your setup)
        self.stm_db = RedisDb(db_url="redis://localhost:6379", expire=time_to_live)

    def memory(self) -> RedisDb:
        return self.stm_db

class LongTermMemory:
    def __init__(self, model_name: str= "BAAI/bge-small-en"):
        self.ltm_db = QdrantDB(db_url="http://localhost:6333", api_key='th3s3cr3tk3y', model_name=model_name)

    def memory(self) -> QdrantDB:
        return self.ltm_db
