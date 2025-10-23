from agno.agent import Agent
from src.semantic_memory.memory_util import ShortTermMemory, LongTermMemory
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
short_term_memory = ShortTermMemory()
long_term_memory = LongTermMemory()
user_id = '7f3a9c2e8b1d4f6a'

# Create agent with Short Term Memory
agent = Agent(
    db=long_term_memory.memory(),
    enable_user_memories=True,
    enable_agentic_memory=True,
)

response_1 = agent.run("Cricket is a good game, I watch cricket for refreshment and recently india won Asia cup 2025", user_id=user_id)
# response_1 = agent.run(input="""
#             My Name is John, i am 45-year-old male patient presenting with:
#             - Persistent fatigue for 3 months
#             - Unexplained weight loss (15 lbs)
#             - Intermittent fever (99-101°F)
#             - Night sweats
#             - Mild abdominal discomfort
#
#             Medical History: Type 2 Diabetes (controlled), Hypertension
#             Current Medications: Metformin 1000mg, Lisinopril 10mg
#             Recent Travel: None
#             Family History: Father had lymphoma at age 60
#
#             These are my problems.
#         """, user_id=user_id)
print(response_1.content)
