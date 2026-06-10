import asyncio
from datetime import datetime, timezone
from typing import AsyncIterator

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text
from rich.table import Table
from rich.rule import Rule
from rich import box
from rich.markup import escape

console = Console()

NODE_LABELS: dict[str, str] = {
    "supervisor": "🧭 Supervisor rou:ting your query",
    "safety_input": "🛡️ Safety-In: checking query",
    "retriever": "🔍 Retriever: searching documents",
    "mcp_tool": "🔧 MCP Tool: querying knowledge base",
    "analyst": "🧠 Analyst: structuring chunks",
    "writer": "✍️ Writer: composing answer",
    "safety_output": "🛡️ Safety-Out: auditing answer",
}

def printBanner() -> None:
    BannerText = Text()
    BannerText.append("Multi-Agent Intelligence Terminal\n", style="bold cyan")
    BannerText.append("LangGraph  ·  Groq  ·  ChromaDB  ·  MCP\n", style="dim white")
    BannerText.append("Type  ", style="dim white")
    BannerText.append("/help", style="bold yellow")
    BannerText.append(" for available commands", style="dim white")
    console.print(Panel(BannerText, border_style="cyan", padding=(1, 4)))

def printHelp() -> None:
    Table_ = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    Table_.add_column("Command", style="bold yellow", no_wrap=True)
    Table_.add_column("Description", style="white")

    Commands = [
        ("/help", "Show this help message"),
        ("/stats", "Query the MCP server for document statistics"),
        ("/files", "List all indexed source files"),
        ("/history", "Show recent query history from long-term memory"),
        ("/prefs", "Show stored user preferences"),
        ("/style concise|detailed", "Set preferred answer style"),
        ("/exit", "Save session report and quit"),
        ("Any text", "Ask a question answered from your indexed documents"),
    ]
    for Cmd, Desc in Commands:
        Table_.add_row(Cmd, Desc)

    console.print(Panel(Table_, title="[bold cyan]Commands[/bold cyan]",
                        border_style="dim cyan", padding=(0, 2)))

def printSeparator() -> None:
    console.print(Rule(style="dim cyan"))

def printNodeProgress(NodeName: str) -> None:
    Label = NODE_LABELS.get(NodeName, f"⚙️   {NodeName}")
    console.print(f"  [dim]{Label}[/dim]")

def printFinalAnswer(Answer: str, CitedSources: list[str] | None = None) -> None:
    AnswerText = Text(escape(Answer))

    if CitedSources:
        SourceLine = "\n\n[dim]Sources:  " + "  ·  ".join(CitedSources) + "[/dim]"
        console.print(
            Panel(
                AnswerText,
                title="[bold green]Answer[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )
        console.print(f"[dim]  📄  Sources: {', '.join(CitedSources)}[/dim]")
    else:
        console.print(
            Panel(
                AnswerText,
                title="[bold green]Answer[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )

def printSafetyBlock(Reason: str, Stage: str = "input") -> None:
    Stage_ = "Query Blocked" if Stage == "input" else "Answer Blocked"
    console.print(
        Panel(
            f"[bold red]{escape(Reason)}[/bold red]",
            title=f"[bold red]⚠️  {Stage_}[/bold red]",
            border_style="red",
            padding=(1, 2),
        )
    )

def printMcpResult(Result: str) -> None:
    console.print(
        Panel(
            escape(Result),
            title="[bold cyan]📊  Knowledge Base[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

def printSessionSummary(SessionId: str, QueryCount: int, BlockedCount: int, SessionNumber: int,) -> None:
    Table_ = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    Table_.add_column("Field", style="dim white", no_wrap=True)
    Table_.add_column("Value", style="white")

    Table_.add_row("Session ID", SessionId)
    Table_.add_row("Session #", str(SessionNumber))
    Table_.add_row("Queries asked", str(QueryCount))
    Table_.add_row("Queries blocked",str(BlockedCount))
    Table_.add_row("Ended at",
                   datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    console.print(
        Panel(
            Table_,
            title="[bold cyan]Session Summary[/bold cyan]",
            border_style="cyan",
            padding=(0, 2),
        )
    )

async def streamGraphResponse(Graph, InitialState: dict, Config: dict) -> dict:
  
    FinalState: dict = {}
    CurrentNode: str = ""
    WriterStreaming: bool = False
    StreamBuffer: list[str] = []

    console.print()  

    try:
        async for Event in Graph.astream_events(InitialState, config=Config, version="v2",):
            EventType = Event.get("event", "")
            EventName = Event.get("name", "")
            EventData = Event.get("data", {})

            if EventType == "on_chain_start" and EventName in NODE_LABELS:
                CurrentNode = EventName
                WriterStreaming = False

                Label = NODE_LABELS[EventName]
                console.print(f"  [dim]{Label}[/dim]")

                if EventName == "writer":
                    WriterStreaming = True
                    StreamBuffer = []
                    console.print("[dim green]Composing...[/dim green]")

            elif EventType == "on_llm_stream" and WriterStreaming:
                Chunk = EventData.get("chunk", {})
                if hasattr(Chunk, "content"):
                    Token = Chunk.content
                elif hasattr(Chunk, "message") and hasattr(Chunk.message, "content"):
                    Token = Chunk.message.content
                else:
                    Token = str(Chunk) if Chunk else ""

                if Token:
                    StreamBuffer.append(Token)
                    console.print(Token, end="", highlight=False)

            elif EventType == "on_chain_end" and EventName in NODE_LABELS:
                if WriterStreaming and StreamBuffer:
                    console.print()  
                    WriterStreaming = False

            elif EventType == "on_chain_end" and EventName == "LangGraph":
                Output = EventData.get("output", {})
                if isinstance(Output, dict):
                    FinalState = Output

    except Exception as Error:
        console.print(f"\n[bold red]Streaming error:[/bold red] {escape(str(Error))}")

    return FinalState

def runStreamingGraph(Graph, InitialState: dict, Config: dict) -> dict:
    return asyncio.run(streamGraphResponse(Graph, InitialState, Config))