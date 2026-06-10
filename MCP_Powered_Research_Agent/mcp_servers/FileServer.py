from pathlib import Path
from mcp.server.fastmcp import FastMCP

McpServer = FastMCP("FileServer")

AllowedDirectory = Path("AllowedFiles").resolve()

@McpServer.tool()
def read_file(FilePath: str) -> str:

    FullPath = (AllowedDirectory / FilePath).resolve()

    if not str(FullPath).startswith(str(AllowedDirectory)):
        return "Access denied."

    if not FullPath.exists():
        return "File not found."

    with open(FullPath, "r", encoding="utf-8") as File:
        return File.read()

if __name__ == "__main__":
    McpServer.run()