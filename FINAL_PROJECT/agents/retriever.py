import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from agents.state import AgentState

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "multi_agent_docs"
TOP_K_RESULTS = 5    


def buildRetrieverVectorStore() -> Chroma:
   
    EmbeddingModel = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    VectorStore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=EmbeddingModel,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return VectorStore

def retrieverNode(State: AgentState) -> dict:
   
    UserQuery = State["UserQuery"]
    print(f"  [Retriever] Searching for: '{UserQuery[:60]}...' " if len(UserQuery) > 60
          else f"  [Retriever] Searching for: '{UserQuery}'")

    VectorStore = buildRetrieverVectorStore()

    DocCount = VectorStore._collection.count()
    if DocCount == 0:
        print("  [Retriever] WARNING: ChromaDB is empty. No documents indexed yet.")
        return {
            "RetrievedChunks": [],
            "RetrievalScores": [],
            "NextAgent": "writer",
            "SessionLog": [{
                "event": "retrieval",
                "query": UserQuery,
                "chunks_found": 0,
            }],
        }

    ResultsWithScores = VectorStore.similarity_search_with_relevance_scores(
        query=UserQuery,
        k=TOP_K_RESULTS,
    )

    RetrievedChunks = []
    RetrievalScores = []
    SourceFiles = []

    for Document, Score in ResultsWithScores:
        RetrievedChunks.append(Document)
        RetrievalScores.append(round(Score, 4))
        SourceFiles.append(Document.metadata.get("source_file", "unknown"))

    print(f"  [Retriever] Found {len(RetrievedChunks)} chunks")
    for Index, (Chunk, Score) in enumerate(zip(RetrievedChunks, RetrievalScores)):
        SourceFile = Chunk.metadata.get("source_file", "unknown")
        print(f"[{Index + 1}] score={Score:.4f}  source={SourceFile}")

    LogEntry = {
        "event": "retrieval",
        "query": UserQuery,
        "chunks_found": len(RetrievedChunks),
        "scores": RetrievalScores,
        "sources": SourceFiles,
    }

    return {
        "RetrievedChunks": RetrievedChunks,
        "RetrievalScores": RetrievalScores,
        "NextAgent": "analyst",
        "SessionLog": [LogEntry],
    }