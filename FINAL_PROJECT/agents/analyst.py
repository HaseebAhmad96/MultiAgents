import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import AgentState

load_dotenv()

GroqApiKey = os.getenv("GroqAPI")

LLM = ChatGroq(
    api_key=GroqApiKey,
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    streaming=True,
)

ANALYST_SYSTEM_PROMPT = """
You are a document analyst. You receive raw text chunks retrieved from a document store.
Your job is to:
1. Identify the most relevant information to the user's query
2. Remove duplicate or redundant content across chunks
3. Organise the key facts into a clean, structured summary
4. Note which source files each fact came from

Output format:
KEY FACTS:
- [fact 1] (source: filename)
- [fact 2] (source: filename)

CONTEXT SUMMARY:
[2-3 sentence summary of what the retrieved chunks collectively say about the query]

GAPS:
[Any important aspect of the query that the retrieved chunks do NOT address]
"""


def analystNode(State: AgentState) -> dict:
  
    UserQuery = State["UserQuery"]
    RetrievedChunks = State.get("RetrievedChunks", [])
    RetrievalScores = State.get("RetrievalScores", [])

    print(f"[Analyst] Structuring {len(RetrievedChunks)} chunks...")

    if not RetrievedChunks:
        return {
            "AnalystSummary": "No relevant documents were found for this query.",
            "NextAgent": "writer",
            "SessionLog": [{
                "event": "analysis",
                "outcome": "no_chunks",
            }],
        }

    ChunksText = ""
    for Index, (Chunk, Score) in enumerate(zip(RetrievedChunks, RetrievalScores)):
        SourceFile = Chunk.metadata.get("source_file", "unknown")
        ChunksText += (
            f"\nChunk {Index + 1}\n"
            f"Source: {SourceFile}\n"
            f"Relevance Score: {Score}\n"
            f"Content:\n{Chunk.page_content}\n"
        )

    AnalystMessages = [
        SystemMessage(content=ANALYST_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"USER QUERY: {UserQuery}\n\n"
            f"RETRIEVED CHUNKS:{ChunksText}"
        )),
    ]

    AnalystResponse = LLM.invoke(AnalystMessages)
    AnalystSummary = AnalystResponse.content.strip()

    print(f"[Analyst] Structured summary ready ({len(AnalystSummary)} chars)")

    LogEntry = {
        "event": "analysis",
        "chunks_analysed": len(RetrievedChunks),
        "summary_length": len(AnalystSummary),
    }

    return {
        "AnalystSummary": AnalystSummary,
        "NextAgent": "writer",
        "SessionLog": [LogEntry],
    }