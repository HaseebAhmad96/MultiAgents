import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from pipeline.document_pipeline import runDocumentPipeline

load_dotenv()

Console = Console()

def verifyEnvironment():
    
    RequiredVars = [
        "GroqAPI",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_PROJECT",
        "CHROMA_PERSIST_DIR",
        "INPUT_DOCS_DIR",
        "MCP_SERVER_HOST",
        "MCP_SERVER_PORT",
        "REPORTS_DIR",
    ]

    MissingVars = []
    for Var in RequiredVars:
        if not os.getenv(Var):
            MissingVars.append(Var)

    if MissingVars:
        Console.print(
            f"[bold red]Missing .env variables:[/bold red] {', '.join(MissingVars)}"
        )
        raise SystemExit(1)

    Console.print("[bold green]✓ Environment verified[/bold green]")

def createDirectories():
    
    Dirs = [
        os.getenv("CHROMA_PERSIST_DIR"),
        os.getenv("INPUT_DOCS_DIR"),
        os.getenv("REPORTS_DIR"),
    ]

    for Dir in Dirs:
        os.makedirs(Dir, exist_ok=True)

    Console.print("[bold green] Directories ready[/bold green]")

def printBanner():

    BannerText = Text()
    BannerText.append("Multi-Agent Intelligence Terminal\n", style="bold cyan")
    BannerText.append("LangGraph · Groq · ChromaDB · MCP\n", style="dim white")
    BannerText.append("Type /help for commands", style="dim white")

    Console.print(Panel(BannerText, border_style="cyan", padding=(1, 4)))

def main():
    printBanner()
    verifyEnvironment()
    createDirectories()
    VectorStore = runDocumentPipeline()

if __name__ == "__main__":
    main()