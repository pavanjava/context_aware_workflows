# Context Aware Workflows

A sophisticated AI agent framework leveraging hybrid memory architecture (short-term Redis + long-term Qdrant) to build context-aware, multi-agent workflows for domain-specific applications.

## Overview

This project demonstrates advanced agentic AI patterns using dual-memory systems where agents maintain conversation context (short-term) while accessing semantic knowledge bases (long-term) for enhanced reasoning and decision-making.

## Architecture

### Memory System
- **Short-Term Memory (STM)**: Redis-based ephemeral storage for conversation context and session state
- **Long-Term Memory (LTM)**: Qdrant vector database with hybrid search (dense + sparse embeddings) for persistent knowledge retrieval

### Key Components
```
context_aware_workflows/
├── src/
│   ├── playgrounds/          # Agent interaction examples
│   │   ├── shared_memory_agent_1.py   # Memory storage agent
│   │   └── shared_memory_agent_2.py   # Memory retrieval agent
│   ├── semantic_memory/      # Memory infrastructure
│   │   ├── memory_util.py    # Memory abstractions
│   │   └── qdrant_db.py      # Hybrid search implementation
│   └── workflows/            # Domain-specific workflows
│       ├── clinical_diagnostic_support.py
│       ├── financial_and_risk_advisory_team.py
│       └── legal_advisory_team.py
```

## Features

### Hybrid Memory Architecture
- **Redis STM**: Fast, expiring cache for active conversations (TTL: 60-120s)
- **Qdrant LTM**: Semantic search using RRF (Reciprocal Rank Fusion) with:
    - Dense embeddings: `BAAI/bge-small-en-v1.5` (384-dim)
    - Sparse embeddings: `prithvida/Splade_PP_en_v1` (IDF-weighted)

### Workflow Patterns

#### 1. Sequential Workflow
**Clinical Diagnostic Support** (`clinical_diagnostic_support.py`)
- Multi-agent medical research team
- Literature review → Clinical guidelines → Diagnostic analysis
- Step-by-step patient case evaluation with memory persistence

#### 2. Conditional Workflow
**Financial Analysis** (`financial_and_risk_advisory_team.py`)
- Dynamic execution paths based on query analysis
- Parallel conditional steps (fundamentals, news, risk)
- Market data → Conditional analysis → Synthesis → Recommendations

#### 3. Parallel Workflow
**Legal Research** (`legal_advisory_team.py`)
- Concurrent case law and statute research
- Parallel research → Legal analysis → Compliance review
- Efficient document generation pipeline

## Installation

### Prerequisites
```bash
# Redis
docker run -d -p 6379:6379 redis:latest

# Qdrant
docker run -d -p 6333:6333 qdrant/qdrant
```

### Setup
```bash
# Clone repository
git clone <repository-url>
cd context_aware_workflows

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
uv pip install -r pyproject.toml

# Configure environment
cp .env.example .env
# Add: OPENAI_API_KEY, DATABASE_URL, QDRANT_API_KEY (optional)
```

## Usage

### Basic Memory Operations
```python
from src.semantic_memory.memory_util import ShortTermMemory, LongTermMemory

# Initialize memory systems
stm = ShortTermMemory(time_to_live=60)
ltm = LongTermMemory()

# Store in long-term memory
ltm.memory().insert(
    text="Important context to remember",
    metadata={"user_id": "user123", "category": "finance"}
)

# Retrieve relevant context
results = ltm.memory().retrieve(query="financial advice", limit=5)
```

### Running Workflows

#### Clinical Diagnostic Support
```bash
python src/workflows/clinical_diagnostic_support.py
```
Analyzes patient symptoms using medical literature and clinical guidelines.

#### Financial Analysis
```bash
python src/workflows/financial_and_risk_advisory_team.py
```
Performs conditional stock analysis with market data, fundamentals, and risk assessment.

#### Legal Research
```bash
python src/workflows/legal_advisory_team.py
```
Conducts parallel legal research for case law and statutes.

## Configuration

### Memory Settings
```python
# Short-term memory TTL (seconds)
short_term_memory = ShortTermMemory(time_to_live=120)

# Long-term memory models
ltm = SemanticLongTermMemory(
    collection_name="long_term_memory",
    sparse_model="prithvida/Splade_PP_en_v1",
    dense_model="BAAI/bge-small-en-v1.5"
)
```

### Agent Configuration
All agents support:
- `enable_user_memories=True`: User-specific memory isolation
- `enable_agentic_memory=True`: Agent reasoning memory
- `user_id`: Memory scoping identifier
- `db`: Memory backend (Redis/Postgres)

## Advanced Features

### Hybrid Search
Combines dense semantic similarity with sparse keyword matching using Reciprocal Rank Fusion for optimal retrieval accuracy.

### Memory Persistence
- Automatic conversation context storage in Redis
- Manual knowledge base updates in Qdrant
- Cross-session memory retrieval by user_id

### Tool Integration
- DuckDuckGo web search
- YFinance financial data
- Google Search for legal research

## Development

### Adding New Workflows
1. Create agent definitions with appropriate tools
2. Define workflow steps (sequential/parallel/conditional)
3. Configure memory backends
4. Implement step input/output transformations

### Memory Management
- STM clears automatically after TTL expiration
- LTM requires manual cleanup or collection management
- Use metadata filtering for efficient retrieval

## Use Cases

- **Healthcare**: Diagnostic support with evidence-based medicine
- **Finance**: Multi-factor investment analysis with risk assessment
- **Legal**: Parallel legal research and memorandum generation
- **Customer Support**: Context-aware responses using conversation history

## Contributing

Contributions welcome for:
- New workflow patterns (hierarchical, graph-based)
- Additional domain implementations
- Memory optimization strategies
- Performance benchmarking

## License

See LICENSE file for details.

## Acknowledgments

Built with:
- [Agno](https://github.com/agno-ai/agno) - Agentic AI framework
- [Qdrant](https://qdrant.tech/) - Vector search engine
- [Redis](https://redis.io/) - In-memory data store
- [FastEmbed](https://github.com/qdrant/fastembed) - Embedding models

## Contact

**Pavan Kumar Mantha**  
- [LinkedIn](https://www.linkedin.com/in/kameshwara-pavan-kumar-mantha-91678b21/) - Profile
- [Medium](https://medium.com/@manthapavankumar11) - Medium Blog 

Talk: "Context Aware Workflows"  
Event: Global AI Hyderabad

---

*This project demonstrates production-ready patterns for building context-aware AI systems with persistent memory and multi-agent orchestration.*