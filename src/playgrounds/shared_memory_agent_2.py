import asyncio

from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv, find_dotenv

from src.semantic_memory.memory_util import ShortTermMemory, LongTermMemory

load_dotenv(find_dotenv())
short_term_memory = ShortTermMemory(time_to_live=120)
long_term_memory = LongTermMemory()
user_id = '7f3a9c2e8b1d4f6a'

# Create agent with Short Term Memory
agent = Agent(
    instructions=("Always make your answer concise and focused to the user query. "
                 "Dont use your prior knowledge. Dont make your answer very generic."
                 "If you dont know just say you 'I dont know'"),
    db=short_term_memory.memory(),
    tools=[DuckDuckGoTools(all=True)],
    enable_user_memories=True,
    enable_agentic_memory=True,
    markdown=True,
    user_id=user_id
)

user_query = 'what is the patient health discussed?'
context_response = long_term_memory.memory().retrieve(query=user_query)

# check for the relevance score before ingesting the docs
# print("\nSearch Results:")
# for result in response:
#     print(f"Score: {result['score']:.4f}")
#     print(f"Text: {result['text']}")
#     print(f"Metadata: {result['metadata']}")
#     print("-" * 50)

# for result in context_response:
#     user_query += f" Context: {result['text']}"

# print(user_query)
# print("="*50)

response = agent.run(input=user_query)
# response = agent.run(input="the talk is at Gachibowli area so can i get some help in getting some places for Lunch near by?", user_id=user_id)
# response = agent.run(input="What is my previous query ?", user_id=user_id)
print(response.content)