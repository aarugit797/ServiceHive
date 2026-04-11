import os
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


def ingest():
    """One-time pipeline: read Markdown KB → chunk → embed → store in ChromaDB."""
    with open("knowledge_base/autostream_kb.md", "r") as f:
        content = f.read()

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "document"),
            ("##", "section"),
        ]
    )
    chunks = splitter.split_text(content)

    print(f"Created {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"  [{chunk.metadata.get('section')}] "
              f"{chunk.page_content[:60]}...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_store"
    )


if __name__ == "__main__":
    ingest()
