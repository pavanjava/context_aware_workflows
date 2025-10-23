from agno.agent import Agent
from dotenv import load_dotenv, find_dotenv

from src.semantic_memory.memory_util import ShortTermMemory, LongTermMemory

load_dotenv(find_dotenv())
short_term_memory = ShortTermMemory()
long_term_memory = LongTermMemory()
user_id = '7f3a9c2e8b1d4f6a'

# Create agent with Short Term Memory
agent = Agent(
    instructions="Always believe in the user input and memories. never use your prior knowledge to make your own decisions. "
                 "Always answer straight to the point",
    db=long_term_memory.memory(),
    enable_user_memories=True,
    enable_agentic_memory=True
)

response = agent.run(input="What is the health condition of the user?")
print(response.content)