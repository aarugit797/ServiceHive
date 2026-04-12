# AutoStream · Aaru — Conversational AI Agent

A production-grade LangGraph AI agent for AutoStream, a fictional SaaS platform for automated video editing. Built with **Streamlit**, **LangGraph**, and **Groq Llama 3.3**.

---

## 🌟 Overview

Aaru is an intelligent conversational agent that handles customer inquiries and captures leads seamlessly using a tightly controlled Directed Acyclic Graph (DAG) state machine. It has completely decoupled from rate-limited, paid APIs by utilizing **Groq** for lightning-fast text generation and **HuggingFace** for local embeddings.

### Key Capabilities
- **Intent Classification Engine:** Routes user messages strictly based on 4 core intents (`greeting`, `inquiry_general`, `inquiry_specific`, `hard_lead`).
- **Local RAG Pipeline:** Retrieves AutoStream pricing, features, and policies via ChromaDB powered by local HuggingFace embeddings (`all-MiniLM-L6-v2`).
- **Conversational Lead Capture:** Extracts Name, Email, and Creator Platform naturally from user chat, asking only for missing fields before triggering a mock tool execution and SQLite storage.
- **Robust Memory:** Multi-turn contextual awareness via LangGraph's `MemorySaver`.

---

## Getting Started

### Prerequisites
- Python 3.10+
- A [Groq API Key](https://console.groq.com/keys) (Free & Instant)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/autostream-agent.git
cd autostream-agent

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
echo "GROQ_API_KEY=your_groq_key_here" > .env

# 5. Ingest knowledge base into ChromaDB (run once)
python ingest.py

# 6. Start the Streamlit application
streamlit run app_streamlit.py
```
The app will automatically open in your browser at `http://localhost:8501`.

---

## 🧠 Architecture Details

### Why LangGraph?
LangGraph's `StateGraph` was chosen over simple LangChain chains because it explicitly manages conversational state across multi-turn exchanges. The graph relies on conditional routing (`route_by_intent` & `route_after_collection`) to enforce strict guardrails ensuring the LLM cannot arbitrarily fire tool executions or hallucinate RAG contexts.

### State Management
Aaru utilizes a multi-layered state management strategy to ensure zero data loss during complex conversations:
- **`AgentState` (TypedDict):** The central source of truth that travels through the LangGraph. it tracks message history, intent history, and the current active node.
- **LangGraph Checkpointing (`MemorySaver`):** Provides short-term working memory. By using a `thread_id` (session ID), Aaru can resume conversations perfectly even after the user refreshes the page or pauses the chat.
- **Pydantic Validation:** The `LeadProfile` within the state uses Pydantic to track missing fields and validation status (`name`, `email`, `platform`).
- **SQLite Persistence:** Once a lead is "captured" by the tool executor, the data is permanently archived in a local SQLite database (`memory/leads.db`) for long-term record keeping.

### Intent Driven Routing
Instead of an LLM guessing what to do, Aaru forces the message through a Llama-3.3 powered Intent Classifier mapped to a strict Pydantic schema. 
- If the user has a **Hard Lead** intent, they are trapped in the Lead Collection node until their profile is complete. 
- If it is an **Inquiry**, the query routes to a standalone RAG node.

### Complete API Independence
This project relies on zero paid APIs. 
- **Language Models**: `ChatGroq` utilizing `llama-3.3-70b-versatile` operating at hundreds of tokens per second.
- **RAG Vector Database**: `HuggingFaceEmbeddings` processing locally, preventing API limits and data leaks.

---

## 📂 File Structure

```text
autostream-agent/
├── knowledge_base/
│   └── autostream_kb.md        ← RAG source (Markdown, split by ## headers)
├── memory/
│   └── db.py                   ← SQLite helpers (init, save_lead, get_all_leads)
├── chroma_store/               ← Auto-created by HuggingFace / ChromaDB on first ingest
├── agent/
│   ├── state.py                ← AgentState TypedDict + Pydantic schemas
│   ├── prompts.py              ← Core system and node prompts
│   ├── tools.py                ← @tool definitions (retrieve_knowledge, mock_lead_capture)
│   ├── nodes.py                ← Graph execution node functions
│   └── graph.py                ← StateGraph orchestration
├── ingest.py                   ← One-time KB → local ChromaDB pipeline
├── app_streamlit.py            ← Main Streamlit chat interface
├── main.py                     ← Terminal fallback chat client
├── requirements.txt
└── .env
```

---

## 🛠️ Testing via CLI
If you prefer not to use the Streamlit interface, you can test the pure agent pipeline directly inside your terminal context:
```bash
python main.py
```
