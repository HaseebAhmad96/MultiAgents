import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from agents.state import AgentState

load_dotenv()

GroqApiKey = os.getenv("GroqAPI")

LLM = ChatGroq(
    api_key=GroqApiKey,
    model="llama-3.3-70b-versatile",
    temperature=0.4,
    streaming=True,
)

WRITER_SYSTEM_PROMPT = """
You are a precise technical writer. Your job is to compose a final answer for the user
based ONLY on the analyst's structured summary of retrieved documents.

Rules you must follow:
1. Only use information present in the analyst summary — never add outside knowledge
2. If the summary says there are gaps, acknowledge them honestly
3. Always cite your sources using: (Source: filename)
4. Write in clear, direct prose — no unnecessary padding
5. End with a SOURCES section listing every file you cited

If the analyst summary says no documents were found, say so clearly and suggest
the user check which files are loaded using the /stats command.
"""


def writerNode(State: AgentState) -> dict:
   
    UserQuery = State["UserQuery"]
    AnalystSummary = State.get("AnalystSummary", "No analysis available.")
    RevisionCount = State.get("RevisionCount", 0)
    PreviousDraft = State.get("FinalAnswer", "")
    SafetyReason = State.get("SafetyReason", "")

    if RevisionCount == 0:
        print(f"[Writer] Composing answer...")
        TaskDescription = "Write a final answer for the user based on the analyst summary below."
    else:
        print(f"[Writer] Revising answer (attempt {RevisionCount + 1})")
        TaskDescription = (
            f"Your previous answer was flagged by the safety agent for this reason:\n"
            f"{SafetyReason}\n\n"
            f"Previous answer:\n{PreviousDraft}\n\n"
            f"Rewrite the answer to fix this issue."
        )

    WriterMessages = [
        SystemMessage(content=WRITER_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"{TaskDescription}\n\n"
            f"USER QUERY: {UserQuery}\n\n"
            f"ANALYST SUMMARY:\n{AnalystSummary}"
        )),
    ]

    WriterResponse = LLM.invoke(WriterMessages)
    FinalAnswer = WriterResponse.content.strip()

    CitedSources = extractCitedSources(FinalAnswer, State.get("RetrievedChunks", []))

    print(f"[Writer] Answer composed ({len(FinalAnswer)} chars)")

    LogEntry = {
        "event": "writing",
        "revision": RevisionCount,
        "answer_length": len(FinalAnswer),
        "cited_sources": CitedSources,
    }

    return {
        "FinalAnswer": FinalAnswer,
        "CitedSources": CitedSources,
        "RevisionCount": RevisionCount + 1,
        "SessionLog": [LogEntry],
    }

def extractCitedSources(AnswerText: str, RetrievedChunks: list) -> list:
   
    CitedSources = []

    AvailableSources = set()
    for Chunk in RetrievedChunks:
        SourceFile = Chunk.metadata.get("source_file", "")
        if SourceFile:
            AvailableSources.add(SourceFile)

    for SourceFile in AvailableSources:
        if SourceFile.lower() in AnswerText.lower():
            CitedSources.append(SourceFile)

    return CitedSources