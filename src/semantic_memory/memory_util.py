from agno.db.redis import RedisDb
from agno.knowledge import Knowledge

from src.semantic_memory.qdrant_db import SemanticLongTermMemory


class ShortTermMemory:
    def __init__(self, time_to_live: int = 60):
        # Setup Redis
        # Initialize Redis db (use the right db_url for your setup)
        self.stm_db = RedisDb(db_url="redis://localhost:6379", expire=time_to_live)

    def memory(self) -> RedisDb:
        return self.stm_db

class LongTermMemory:
    def __init__(self):
        # model_name can be "snowflake/snowflake-arctic-embed-m"
        self.ltm_db = SemanticLongTermMemory()

    def memory(self) -> SemanticLongTermMemory:
        return self.ltm_db
