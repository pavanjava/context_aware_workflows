import asyncio
from textwrap import dedent

from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from src.semantic_memory.memory_util import ShortTermMemory, LongTermMemory
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())
short_term_memory = ShortTermMemory()
long_term_memory = LongTermMemory()
user_id = '7f3a9c2e8b1d4f6a'

# Create agent with Short Term Memory
agent = Agent(
    name="general personal Assistant",
    description="You are a generic personal assistant who can search web and check memories to answer users questions.",
    instructions=dedent("""\
        Always check the answer factually before giving it to user. 
        Always try to use your search tools to get the answer more relevant with most recent information. 
        Never use your prior knowledge.
        Always check if the memory has any relevant answer previously.\
        """),
    db=short_term_memory.memory(),
    tools=[DuckDuckGoTools(all=True)],
    enable_user_memories=True,
    enable_agentic_memory=True,
    markdown=True,
    user_id=user_id
)

response_1 = agent.run("My name is pavan and tomorrow i am speaking on 'context aware workflows' organised by Global AI Hyderabad. ", user_id=user_id)
# response_1 = agent.run("What is my previous question and what is its answer?", user_id=user_id)
# response_1 = agent.run("I am planning to stay back at Gachibowli after the talk so, can i get some good hotels (hotel name) to stay along with address and rating", user_id=user_id)
long_term_memory.memory().insert(text=response_1.content, metadata={"user_id": user_id})
print(response_1.content)
