import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

MCP_HOST = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_SERVER_PORT", "8765"))

Mcp = FastMCP("DocumentIntelligenceServer")

@Mcp.tool()
def get_document_stats() -> str:
   
    try:
        from pipeline.document_pipeline import getDocumentStats
        Stats  = getDocumentStats()
        return json.dumps(Stats, indent=2)
    except Exception as Error:
        return json.dumps({"error": str(Error)})

@Mcp.tool()
def list_source_files() -> str:
    
    try:
        from pipeline.document_pipeline import getDocumentStats
        Stats = getDocumentStats()
        Files = Stats.get("source_files", [])
        if not Files:
            return "No files currently indexed in the vector store."
        return "Indexed files:\n" + "\n".join(f"- {F}" for F in Files)
    except Exception as Error:
        return f"Error retrieving file list: {str(Error)}"

@Mcp.tool()
def search_metadata(keyword: str) -> str:
   
    try:
        from pipeline.document_pipeline import getDocumentStats
        Stats = getDocumentStats()
        Files = Stats.get("source_files", [])
        Matches = [F for F in Files if keyword.lower() in F.lower()]

        if not Matches:
            return f"No indexed files matching '{keyword}'."
        return f"Files matching '{keyword}':\n" + "\n".join(f"- {M}" for M in Matches)
    except Exception as Error:
        return f"Error searching metadata: {str(Error)}"

if __name__ == "__main__":
    print(f"[MCP] Starting DocumentIntelligenceServer on {MCP_HOST}:{MCP_PORT}")
    Mcp.run(transport="streamable-http", host=MCP_HOST, port=MCP_PORT)