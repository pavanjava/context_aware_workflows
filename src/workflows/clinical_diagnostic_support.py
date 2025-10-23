import asyncio
import os
from textwrap import dedent

from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.models.openai import OpenAIChat
from agno.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.workflow.types import StepInput, StepOutput
from agno.workflow.workflow import Workflow
from dotenv import load_dotenv, find_dotenv

from src.semantic_memory.memory_util import ShortTermMemory, LongTermMemory

load_dotenv(find_dotenv())
db = PostgresDb(db_url=os.environ.get("DATABASE_URL"))

short_term_memory = ShortTermMemory()
long_term_memory = LongTermMemory()
user_id = '7f3a9c2e8b1d4f6a'

# Define agents
medical_literature_agent = Agent(
    name="Medical Literature Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[DuckDuckGoTools()],
    role="Search for peer-reviewed medical research, clinical trials, and latest treatment protocols",
    db=long_term_memory.memory(),
    enable_user_memories=True,
    enable_agentic_memory=True
)

clinical_guidelines_agent = Agent(
    name="Clinical Guidelines Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    # tools=[DuckDuckGoTools()],  # Replace with medical database tools if available
    role="Extract evidence-based guidelines, dosage protocols, and contraindications from medical databases",
    db=long_term_memory.memory(),
    enable_user_memories=True,
    enable_agentic_memory=True
)

diagnostic_specialist_agent = Agent(
    name="Diagnostic Specialist Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="Analyze patient symptoms, lab results, and medical history to provide differential diagnosis recommendations",
    db=long_term_memory.memory(),
    enable_user_memories=True,
    enable_agentic_memory=True
)


async def prepare_patient_case_input(step_input: StepInput) -> StepOutput:
    patient_symptoms = step_input.input
    return StepOutput(
        content=dedent(f"""\
	Patient presenting with the following symptoms and concerns:
	<patient_case>
	{patient_symptoms}
	</patient_case>

	Search for:
	1. Latest clinical research on these symptoms
	2. Current treatment protocols and guidelines
	3. Recent case studies with similar presentations
	4. Drug interactions and contraindications
	5. Evidence-based diagnostic criteria
	
	Retrieve at least 10 relevant medical sources\
	""")
    )


async def prepare_diagnostic_report_input(step_input: StepInput) -> StepOutput:
    patient_symptoms = step_input.input
    research_findings = step_input.previous_step_content

    return StepOutput(
        content=dedent(f"""\
	Generate a comprehensive diagnostic analysis report for:
	<patient_case>
	{patient_symptoms}
	</patient_case>

	Based on the following medical research and clinical guidelines:
	<clinical_research>
	{research_findings}
	</clinical_research>
	
	Provide:
	1. Differential diagnosis with probability rankings
	2. Recommended diagnostic tests and procedures
	3. Evidence-based treatment options
	4. Potential complications and red flags
	5. Follow-up care recommendations
	6. Patient education points\
	""")
    )


# Define medical research team
medical_research_team = Team(
    name="Medical Research Team",
    members=[clinical_guidelines_agent, medical_literature_agent],
    instructions="Conduct comprehensive medical literature review and extract clinical guidelines for patient case analysis",
)


# Create diagnostic workflow
if __name__ == "__main__":
    clinical_diagnosis_workflow = Workflow(
        name="Clinical Diagnostic Support Workflow",
        description="AI-assisted diagnostic analysis using latest medical research and clinical guidelines",
        # db=PostgresDb(session_table="clinical_workflow_session", db_url=os.environ.get("DATABASE_URL")),
        steps=[
            prepare_patient_case_input,
            medical_research_team,
            prepare_diagnostic_report_input,
            diagnostic_specialist_agent,
        ],
        db=long_term_memory.memory(),
    )

    # response = clinical_diagnosis_workflow.run(
    #     input="""
    #         45-year-old male patient presenting with:
    #         - Persistent fatigue for 3 months
    #         - Unexplained weight loss (15 lbs)
    #         - Intermittent fever (99-101°F)
    #         - Night sweats
    #         - Mild abdominal discomfort
    #
    #         Medical History: Type 2 Diabetes (controlled), Hypertension
    #         Current Medications: Metformin 1000mg, Lisinopril 10mg
    #         Recent Travel: None
    #         Family History: Father had lymphoma at age 60
    #     """,
    #     user_id=user_id
    # )
    #
    # print(response.content)

    asyncio.run(
        clinical_diagnosis_workflow.aprint_response(
            input="""
            As a doctor i want you to analyse and give me the report 
            Patient Details:
            the Patient 45-year-old male presenting with:
            - Persistent fatigue for 3 months
            - Unexplained weight loss (15 lbs)
            - Intermittent fever (99-101°F)
            - Night sweats
            - Mild abdominal discomfort

            Medical History: Type 2 Diabetes (controlled), Hypertension
            Current Medications: Metformin 1000mg, Lisinopril 10mg
            Recent Travel: None
            Family History: Father had lymphoma at age 60
            """,
            markdown=True,
            user_id=user_id,
        )
    )