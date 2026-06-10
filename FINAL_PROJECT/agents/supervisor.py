import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import AgentState

load_dotenv()

GroqApiKey = os.getenv("GroqAPI")

Llm = ChatGroq(
    api_key=GroqApiKey,
    model="llama-3.3-70b-versatile",
    temperature=0,     
    streaming=True,
)

SUPERVISOR_SYSTEM_PROMPT = """
You are a supervisor agent that routes user queries to the correct worker.

Available workers:
- retriever  : Use when the query needs information from the loaded documents
- mcp_tool   : Use when the user asks about document stats, which files are loaded,
               or anything about the knowledge base itself
- end        : Use when the query is a greeting, small talk, or needs no document lookup

Respond with ONLY one word — the worker name. Nothing else.
Examples:
  User: "What does the report say about revenue?"  → retriever
  User: "Which files are loaded?"                  → mcp_tool
  User: "How many documents do I have?"            → mcp_tool
  User: "Hello"                                    → end
  User: "Thanks"                                   → end
"""


def supervisorNode(State: AgentState) -> dict:
   
    UserQuery = State["UserQuery"]

    print(f"\n  [Supervisor] Routing query...")

    RoutingMessages = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=UserQuery),
    ]

    RoutingResponse = Llm.invoke(RoutingMessages)
    Decision = RoutingResponse.content.strip().lower()

    Decision = Decision.replace(".", "").replace(",", "").strip()

    ValidDecisions = {"retriever", "mcp_tool", "end"}
    if Decision not in ValidDecisions:
        Decision = "retriever"

    print(f"  [Supervisor] -> Routing to: {Decision}")

    LogEntry = {
        "event": "supervisor_routing",
        "query": UserQuery,
        "decision": Decision,
    }

    return { "NextAgent":  Decision, "SessionLog": [LogEntry], }

def mcpToolNode(State: AgentState) -> dict:
  
    import httpx
    import json

    MCP_HOST = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    MCP_PORT = os.getenv("MCP_SERVER_PORT", "8765")

    UserQuery = State["UserQuery"].lower()

    if "which files" in UserQuery or "list" in UserQuery or "what files" in UserQuery:
        ToolName = "list_source_files"
        ToolParams = {}
    elif "stat" in UserQuery or "how many" in UserQuery or "count" in UserQuery:
        ToolName = "get_document_stats"
        ToolParams = {}
    else:
        ToolName = "get_document_stats"
        ToolParams = {}

    print(f"  [MCP] Calling tool: {ToolName}")

    try:
        Response = httpx.post(
            f"http://{MCP_HOST}:{MCP_PORT}/tools/{ToolName}",
            json=ToolParams,
            timeout=10.0,
        )
        ToolResult = Response.text

    except Exception as Error:
        ToolResult = f"MCP server unavailable: {str(Error)}. Is mcp_server/server.py running?"

    LogEntry = {
        "event":  "mcp_tool_call",
        "tool":   ToolName,
        "result": ToolResult[:200],
    }

    return {
        "McpToolResult": ToolResult,
        "FinalAnswer": ToolResult,
        "NextAgent": "end",
        "SessionLog": [LogEntry],
    }

def supervisorRouter(State: AgentState) -> str:
  
    NextAgent = State.get("NextAgent", "end")

    RouteMap = {
        "retriever": "safety_input",   
        "mcp_tool": "mcp_tool",
        "end": "end",
    }

    return RouteMap.get(NextAgent, "end")