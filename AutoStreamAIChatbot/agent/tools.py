import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()



embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="./chroma_store",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


@tool
def retrieve_knowledge(query: str) -> str:
    """
    Retrieves relevant AutoStream product information, pricing,
    and policy details from the local knowledge base. Use whenever
    the user asks about features, pricing, plans, or policies.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in knowledge base."
    results = []
    for doc in docs:
        section = doc.metadata.get("section", "General")
        results.append(f"[{section}]\n{doc.page_content}")
    return "\n\n---\n\n".join(results)


@tool
def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Captures a qualified lead. Only call this once name, email,
    and creator platform are ALL confirmed. Never call prematurely.
    """
    print(f"Lead captured successfully: {name}, {email}, {platform}")
    return f"Lead captured successfully: {name}, {email}, {platform}"
