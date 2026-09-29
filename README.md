# 🛒 AutoGen + RAG for E-Commerce — Multi-Agent Retrieval, Tool Calling & Answer Synthesis (Agentic AI #7)

Where this series' two threads finally meet: the **RAG concepts** from the GenAI series (retrieval grounding, vector search, honest refusal) built entirely inside **AutoGen's multi-agent architecture**. Two independent ChromaDB-backed retrieval crews — one for products, one for customer orders — search their own collections via AutoGen's native **function-calling** mechanism, run concurrently through **async group chats**, and hand their findings to a dedicated **writer agent** that synthesises the final answer from retrieved data alone.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![AutoGen](https://img.shields.io/badge/Microsoft-AutoGen-0078D4?logo=microsoft&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Search-FF6F00)
![Agentic AI](https://img.shields.io/badge/Series-Agentic%20AI%20%2307-8A2BE2)

---

## 📌 Overview

Every earlier RAG notebook in this portfolio built retrieval around a LangChain chain. This one asks a different question: **what does RAG look like when agents, not chains, do the retrieving?** Two full retrieval pipelines — products and orders — are each built as a small AutoGen agent team with a registered search tool, run **concurrently and asynchronously**, and their combined findings are handed off to a separate synthesis agent for the final grounded answer.

---

## 🏗️ The Complete Pipeline

```
products.csv ──► products_collection (ChromaDB)     orders.csv ──► orders_collection (ChromaDB)
        │                                                    │
        ▼                                                    ▼
products_search_assistant_agent                    orders_search_assistant_agent
  (tool: search_products, registered via                (tool: search_orders, registered via
   @register_for_llm / @register_for_execution)           @register_for_llm / @register_for_execution)
        │                                                    │
        ▼                                                    ▼
products_groupchat (round_robin)                    orders_groupchat (round_robin)
        │                                                    │
        └──────────────────┬─────────────────────────────────┘
                            ▼
              a_initiate_chats() ──► BOTH run asynchronously, concurrently
                            │
                            ▼
              retrieved_data (combined chat history from both crews)
                            │
                            ▼
              writer_assistant_agent ──► final grounded answer
```

---

## 🔬 Part 1 — Two Independent ChromaDB Collections

```python
chroma_client = chromadb.Client()
products_collection = chroma_client.create_collection(name="products")
orders_collection = chroma_client.create_collection(name="orders")

with open("/content/products.csv") as file:
    for line in csv.reader(file):
        if line[0] != "Product Name":
            products_collection.add(documents=str(line), ids=str(id))
            id += 1
```
Each CSV row becomes one Chroma document. Keeping **products** and **orders** as entirely separate collections — rather than one mixed knowledge base — means each retrieval agent only ever searches the data relevant to its own domain, a clean separation that scales naturally as more data sources are added.

---

## 🔬 Part 2 — AutoGen's Native Tool-Calling Pattern

```python
@products_search_executor_agent.register_for_execution()
@products_search_assistant_agent.register_for_llm(
    description="Search a ChromaDB collection containing information about products.")
async def search_products(search_query: Annotated[str, "Search query"]):
    results = products_collection.query(query_texts=[search_query], n_results=2)
    return results["documents"]
```
This is a genuinely different tool-calling mechanism from Part 6's LangGraph `bind_tools()` + `ToolNode` pattern. AutoGen splits the responsibility across **two decorators on two different agents**: `register_for_llm` tells the *assistant* agent this function exists and how to describe it when deciding to call it; `register_for_execution` tells the *executor* agent it's the one actually allowed to run it. Reasoning and execution are deliberately separated onto different agents, not just different graph nodes.

---

## 🔬 Part 3 — Round-Robin Group Chats (A Third Speaker-Selection Strategy)

```python
products_groupchat = autogen.GroupChat(
    agents=[products_search_assistant_agent, products_search_executor_agent],
    messages=[], max_round=8, speaker_selection_method="round_robin",
)
```
Part 3 of this series covered `'auto'` (LLM-decided) and a custom `state_transition` function (fully deterministic, content-aware) speaker selection. `"round_robin"` is a third strategy: agents simply take turns **in the order they were listed**, no LLM decision and no custom logic required — the right choice here, since a two-agent search-and-execute loop has an obviously fixed turn order.

---

## 🔬 Part 4 — Running Both Retrieval Crews Concurrently, Asynchronously

```python
async_chat_plan = [
    {"chat_id": 1, "recipient": products_groupchat_manager, "message": "What is Artisanal Air?.",
     "summary_method": "reflection_with_llm", "silent": False},
    {"chat_id": 2, "recipient": orders_groupchat_manager, "message": "What did Ned Noodle order?",
     "summary_method": "reflection_with_llm", "silent": False},
]

async def start_groupchat():
    user = autogen.UserProxyAgent(name="User", human_input_mode="NEVER", ...)
    await user.a_initiate_chats(async_chat_plan)

await start_groupchat()
```
`a_initiate_chats()` — the **async** counterpart to Part 2's `initiate_chats()` — runs both retrieval crews concurrently rather than strictly sequentially, a real efficiency gain when multiple independent lookups don't depend on each other's results.

---

## 🔬 Part 5 — A Dedicated Writer Agent for Final Synthesis

```python
writer_assistant_agent = autogen.AssistantAgent(
    name="writer_assistant",
    system_message="""You are a helpful assistant for a company.
Your job is to answer the user's question using the provided information.
DO NOT rely on your own knowledge, ONLY use the provided info.
If you don't know the answer, just say you don't know.""",
    ...
)

writer_prompt = f"""Please write the final answer to the user's question: {user_question}
The information retrieved from the search agents is: {retrieved_data}."""
```
This mirrors the exact retrieval-then-generation split at the heart of every RAG notebook earlier in this portfolio — but here it's implemented as **two AutoGen agent teams handing off to a third agent**, rather than a LangChain retriever piping into a prompt template. The explicit grounding instructions ("DO NOT rely on your own knowledge... if you don't know, say so") continue the same honesty discipline tested throughout this whole body of work.

---

## 🗂️ Repository Structure

```
autogen-rag-ecommerce/
├── Autogen_RAG_Tutorial_for_ecommerce.ipynb   # Main notebook
├── products.csv                                 # Sample product catalog
├── orders.csv                                   # Sample customer order data
├── requirements.txt                             # Dependencies
├── .gitignore                                   # Keeps secrets, checkpoints & caches out of git
├── .env.example                                 # Template for required environment variables
└── README.md                                    # This documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/api-keys) with `gpt-4o` access

### Installation

```bash
git clone https://github.com/Kailaswadje/autogen-rag-ecommerce.git
cd autogen-rag-ecommerce

pip install -r requirements.txt

jupyter notebook Autogen_RAG_Tutorial_for_ecommerce.ipynb
```

> ⚠️ The notebook already uses `getpass()` for the API key — keep it. Update the hardcoded `/content/products.csv` and `/content/orders.csv` paths for a local environment. `chromadb.Client()` here is **in-memory only** — data resets every run, which is fine for this demo but worth knowing if you extend it. Clear notebook outputs before pushing.

---

## 🧠 Key Takeaways

- **RAG's core idea — retrieve, then generate from retrieved context only — is framework-independent** — this notebook proves the same pattern built earlier with LangChain chains works just as well as a set of coordinating AutoGen agents
- **AutoGen splits tool "knowledge" and tool "execution" across two agents**, via `register_for_llm` and `register_for_execution` — a genuinely different design from LangGraph's single-node `ToolNode`
- **`round_robin` is the right speaker-selection choice for a fixed, predictable turn order** — not every group chat needs `'auto'` or a custom state machine
- **`a_initiate_chats()` runs independent agent crews concurrently** — a real efficiency gain when retrieval tasks don't depend on each other
- **Separating retrieval agents from a dedicated writer/synthesis agent** is the same architectural split every RAG chain in this portfolio uses, just implemented with agents instead of a LangChain pipeline
- This crossover between the RAG techniques and the multi-agent orchestration patterns covered separately elsewhere in this portfolio is a direct rehearsal for the retrieval-and-synthesis agents in my dissertation's agentic intelligence platform

---

## 📚 Agentic AI Series Context

| Part | Project | Framework | Focus |
|---|---|---|---|
| 01 | [AutoGen Agent Fundamentals](https://github.com/Kailaswadje/agentic-ai-autogen-introduction) | AutoGen | `ConversableAgent`, peer-to-peer negotiation |
| 02 | [UserProxyAgent & Sequential Chat](https://github.com/Kailaswadje/agentic-ai-userproxyagent-sequential-chat) | AutoGen | Human-facing coordination, pipeline handoffs |
| 03 | [Group Chat, State Flow & Nested Chat](https://github.com/Kailaswadje/agentic-ai-group-chat-state-flow-nested-chat) | AutoGen | Multi-agent teams, deterministic orchestration |
| 04 | [CrewAI Fundamentals](https://github.com/Kailaswadje/crewai-fundamentals-recipe-crew) | CrewAI | Structured agents, task/agent separation, planning |
| 05 | [CrewAI with Real Web Search](https://github.com/Kailaswadje/crewai-web-search-market-research-crew) | CrewAI | Tool-equipped agents, automatic sequential context |
| 06 | [LangGraph Fundamentals](https://github.com/Kailaswadje/langgraph-fundamentals-react-agent) | LangGraph | Explicit state graphs, ReAct tool loop |
| **07 (this repo)** | AutoGen + RAG for E-Commerce | AutoGen | Agent-based retrieval, native tool calling, async orchestration |

---

## 🔮 Possible Extensions

- [ ] Persist ChromaDB to disk (`PersistentClient`) instead of the current in-memory `Client()`
- [ ] Add a third collection/crew (e.g. shipping status) and extend `async_chat_plan` accordingly
- [ ] Replace `round_robin` with a custom `state_transition` function if the search-and-verify loop needs conditional logic
- [ ] Fix the `documents=str(line)` / `ids=str(id)` calls to pass proper single-item lists for stricter ChromaDB API compliance

---

## 👤 Author

**Kailas Wadje**
MSc Data Science & AI, University of Liverpool

- GitHub: [@Kailaswadje](https://github.com/Kailaswadje)
- LinkedIn: [linkedin.com/in/kwadaje](https://www.linkedin.com/in/kwadaje/)

---

⭐ If seeing RAG rebuilt entirely on multi-agent tool calling connected the dots for you, consider giving it a star!
