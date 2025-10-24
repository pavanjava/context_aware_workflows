import asyncio
from datetime import datetime

from agno.agent.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.workflow.condition import Condition
from agno.workflow.parallel import Parallel
from agno.workflow.step import Step
from agno.workflow.types import StepInput
from agno.workflow.workflow import Workflow

from src.semantic_memory.memory_util import ShortTermMemory, LongTermMemory
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Initialize memory
short_term_memory = ShortTermMemory()
long_term_memory = LongTermMemory()
user_id = '7f3a9c2e8b1d4f6a'

# === FINANCIAL AGENTS ===
market_data_agent = Agent(
    name="Market Data Analyst",
    instructions="""You are a market data analyst. Your role is to:
    - Fetch real-time stock prices, financial metrics, and market data
    - Analyze historical price trends and trading volumes
    - Provide technical analysis insights
    Always cite the data sources and timestamps.""",
    tools=[YFinanceTools()],
    db=short_term_memory.memory(),
    enable_user_memories=True,
    enable_agentic_memory=True,
    user_id=user_id,
)

news_analyst_agent = Agent(
    name="Financial News Analyst",
    instructions="""You are a financial news analyst. Your role is to:
    - Search for latest financial news and earnings reports
    - Identify market-moving events and sentiment
    - Analyze impact of news on stock performance
    Focus on recent and relevant information.""",
    tools=[DuckDuckGoTools()],
    db=short_term_memory.memory(),
    enable_user_memories=True,
    enable_agentic_memory=True,
    user_id=user_id,
)

fundamental_analyst_agent = Agent(
    name="Fundamental Analyst",
    instructions="""You are a fundamental analyst. Your role is to:
    - Analyze company financials (P/E, EPS, revenue, profit margins)
    - Evaluate business model and competitive advantages
    - Assess valuation metrics
    Provide data-driven insights.""",
    tools=[YFinanceTools()],
    db=short_term_memory.memory(),
    enable_user_memories=True,
    enable_agentic_memory=True,
    user_id=user_id,
)

risk_analyst_agent = Agent(
    name="Risk Analyst",
    instructions="""You are a risk analyst. Your role is to:
    - Assess volatility and beta metrics
    - Identify potential risks (market, sector, company-specific)
    - Evaluate diversification needs
    - Calculate risk-adjusted returns
    Be conservative and highlight all major risks.""",
    tools=[YFinanceTools()],
    db=short_term_memory.memory(),
    enable_user_memories=True,
    enable_agentic_memory=True,
    user_id=user_id,
)

portfolio_strategist_agent = Agent(
    name="Portfolio Strategist",
    instructions="""You are a portfolio strategist. Your role is to:
    - Synthesize all research data into actionable insights
    - Provide investment recommendations with clear reasoning
    - Suggest position sizing and entry/exit strategies
    - Create comprehensive investment summaries
    Balance growth potential with risk management.""",
    db=short_term_memory.memory(),
    enable_user_memories=True,
    enable_agentic_memory=True,
    user_id=user_id,
)

# === WORKFLOW STEPS ===
market_data_step = Step(
    name="FetchMarketData",
    description="Fetch current market data and price information",
    agent=market_data_agent,
)

news_analysis_step = Step(
    name="AnalyzeNews",
    description="Analyze recent financial news and sentiment",
    agent=news_analyst_agent,
)

fundamental_analysis_step = Step(
    name="FundamentalAnalysis",
    description="Perform fundamental analysis of financials",
    agent=fundamental_analyst_agent,
)

risk_assessment_step = Step(
    name="AssessRisk",
    description="Assess investment risks and volatility",
    agent=risk_analyst_agent,
)

synthesize_research_step = Step(
    name="SynthesizeResearch",
    description="Combine all research findings",
    agent=portfolio_strategist_agent,
)

generate_recommendation_step = Step(
    name="GenerateRecommendation",
    description="Create final investment recommendation in the form of time article",
    agent=portfolio_strategist_agent,
)


# === CONDITION EVALUATORS ===
def should_analyze_stock_fundamentals(step_input: StepInput) -> bool:
    """Check if we should perform fundamental analysis"""
    query = step_input.input or step_input.previous_step_content or ""
    fundamental_keywords = [
        "valuation", "earnings", "revenue", "profit",
        "fundamental", "financials", "balance sheet", "p/e"
    ]
    return any(keyword in query.lower() for keyword in fundamental_keywords)


def should_analyze_market_news(step_input: StepInput) -> bool:
    """Check if we should analyze news"""
    query = step_input.input or step_input.previous_step_content or ""
    news_keywords = [
        "news", "announcement", "earnings report", "sentiment",
        "market", "update", "latest"
    ]
    return any(keyword in query.lower() for keyword in news_keywords)


def should_perform_risk_analysis(step_input: StepInput) -> bool:
    """Check if we should perform risk analysis"""
    query = step_input.input or step_input.previous_step_content or ""
    risk_keywords = [
        "risk", "volatility", "beta", "downside",
        "safe", "conservative", "diversification"
    ]
    return any(keyword in query.lower() for keyword in risk_keywords)


if __name__ == "__main__":
    # Create the financial analysis workflow
    workflow = Workflow(
        name="Financial Analysis Workflow",
        description="Comprehensive financial analysis and investment research workflow",
        steps=[
            # Step 1: Always fetch current market data first
            market_data_step,

            # Step 2: Run parallel conditional analysis based on query
            Condition(
                name="FundamentalAnalysisCondition",
                description="Perform fundamental analysis if requested",
                evaluator=should_analyze_stock_fundamentals,
                steps=[fundamental_analysis_step],
            ),
            Condition(
                name="NewsAnalysisCondition",
                description="Analyze news if relevant",
                evaluator=should_analyze_market_news,
                steps=[news_analysis_step],
            ),
            Condition(
                name="RiskAnalysisCondition",
                description="Perform risk analysis if requested",
                evaluator=should_perform_risk_analysis,
                steps=[risk_assessment_step],
            ),

            # Step 3: Synthesize all findings
            synthesize_research_step,

            # Step 4: Generate final recommendation
            generate_recommendation_step,
        ],
    )

    # Example queries to test different paths
    test_queries = [
        "Analyze AAPL stock with fundamental analysis and risk assessment",
        "What's the latest news on Tesla stock performance?",
        "Should I invest in NVDA? Check valuation and risk",
        "Get me current market data for Microsoft",
    ]

    # Run the workflow with a sample query
    try:
        print(f"{'=' * 60}")
        print(f"Financial Analysis Workflow - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}\n")

        response = asyncio.run(
            workflow.arun(
                input="Analyze AAPL stock with fundamental and Risk analysis and news sentiment",
                user_id=user_id,
            )
        )

        print(response.content)
        long_term_memory.memory().insert(text=response.content, metadata={'user_id': user_id})

        print(f"\n{'=' * 60}")
        print("Workflow completed successfully!")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"❌ Error executing workflow: {e}")
        import traceback

        traceback.print_exc()
