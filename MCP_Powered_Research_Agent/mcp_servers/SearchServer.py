from mcp.server.fastmcp import FastMCP
import json

McpServer = FastMCP("SearchServer")

@McpServer.tool()
def search_web(Query: str) -> str:
    
    with open("search_data.json", "r") as File:
        SearchData = json.load(File)

    Query = Query.lower()

    for Key, Value in SearchData.items():
        if Key in Query:
            return "\n".join(Value)

    return "No search results found."

if __name__ == "__main__":
    McpServer.run()