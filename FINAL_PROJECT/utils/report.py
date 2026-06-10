import os
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPORTS_DIR = os.getenv("REPORTS_DIR", "./reports")

def writeSessionReport(SessionId: str, SessionLog: list[dict], SafetyDecisions: list[dict], UserPreferences: dict, SessionNumber: int,) -> tuple[str, str]:
   
    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)

    Timestamp = datetime.now(timezone.utc)
    TimestampStr = Timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")

    QAPairs = _extractQAPairs(SessionLog)
    Retrievals = _extractByEvent(SessionLog, "retrieval")
    SafeBlocked = [D for D in SafetyDecisions if D.get("verdict") == "UNSAFE"]

    MarkdownContent = _buildMarkdown(
        SessionId, TimestampStr, SessionNumber,
        QAPairs, SafetyDecisions, SafeBlocked,
        Retrievals, UserPreferences,
    )

    JsonContent = _buildJson(
        SessionId, TimestampStr, SessionNumber,
        QAPairs, SafetyDecisions,
        Retrievals, UserPreferences,
        SessionLog,
    )

    MdPath   = os.path.join(REPORTS_DIR, f"{SessionId}.md")
    JsonPath = os.path.join(REPORTS_DIR, f"{SessionId}.json")

    with open(MdPath, "w", encoding="utf-8") as F:
        F.write(MarkdownContent)

    with open(JsonPath, "w", encoding="utf-8") as F:
        json.dump(JsonContent, F, indent=2, ensure_ascii=False)

    return MdPath, JsonPath

def _buildMarkdown(SessionId, TimestampStr, SessionNumber, QAPairs, SafetyDecisions, SafeBlocked, Retrievals, UserPreferences,) -> str:

    Lines = []

    Lines += [
        f"# Session Report — {SessionId}",
        f"",
        f"| Field            | Value                        |",
        f"|------------------|------------------------------|",
        f"| Session ID       | `{SessionId}`                |",
        f"| Session Number   | {SessionNumber}              |",
        f"| Generated        | {TimestampStr}               |",
        f"| Total Queries    | {len(QAPairs)}               |",
        f"| Blocked Queries  | {len(SafeBlocked)}           |",
        f"",
    ]

    Lines += ["## User Preferences", ""]
    if UserPreferences:
        for K, V in UserPreferences.items():
            Lines.append(f"- **{K}**: {V}")
    else:
        Lines.append("_No preferences stored._")
    Lines.append("")

    Lines += ["## Questions & Answers", ""]
    if QAPairs:
        for Idx, QA in enumerate(QAPairs, 1):
            Query = QA.get("query", "—")
            Answer = QA.get("answer", "_No answer generated_")
            Sources = QA.get("cited_sources", [])
            InputSafe = QA.get("input_safe", True)
            OutputSafe = QA.get("output_safe", True)
            Revision = QA.get("revision", 0)

            SafetyBadge = ""
            if not InputSafe:
                SafetyBadge = "  ⚠️ **[INPUT BLOCKED]**"
            elif not OutputSafe:
                SafetyBadge = "  ⚠️ **[OUTPUT BLOCKED]**"

            Lines += [
                f"### Q{Idx}.  {Query}{SafetyBadge}",
                f"",
                f"**Answer:**",
                f"",
                f"{Answer}",
                f"",
            ]

            if Sources:
                Lines.append(f"**Sources cited:** {', '.join(f'`{S}`' for S in Sources)}")
                Lines.append("")

            if Revision > 1:
                Lines.append(f"_Answer required {Revision} revision(s) before passing safety._")
                Lines.append("")

            Lines.append("---")
            Lines.append("")
    else:
        Lines.append("_No questions were asked this session._")
        Lines.append("")

    Lines += ["## Safety Audit Trail", ""]
    if SafetyDecisions:
        for D in SafetyDecisions:
            Stage = D.get("stage", "?")
            Verdict = D.get("verdict", "?")
            Reason = D.get("reason", "")
            Marker = "✅" if Verdict == "SAFE" else "🚫"
            Lines.append(f"- {Marker}  **[{Stage.upper()}]**  {Verdict}  —  {Reason}")
    else:
        Lines.append("_No safety checks recorded._")
    Lines.append("")

    Lines += ["## Retrieval Statistics", ""]
    if Retrievals:
        TotalChunks = sum(R.get("chunks_found", 0) for R in Retrievals)
        Lines.append(f"- Total retrieval calls: **{len(Retrievals)}**")
        Lines.append(f"- Total chunks retrieved across all queries: **{TotalChunks}**")
        Lines.append("")
        Lines.append("| # | Query | Chunks Found | Top Score |")
        Lines.append("|---|-------|--------------|-----------|")
        for Idx, R in enumerate(Retrievals, 1):
            Query_ = R.get("query", "—")[:60]
            Chunks = R.get("chunks_found", 0)
            Scores = R.get("scores", [])
            TopScore = f"{max(Scores):.4f}" if Scores else "—"
            Lines.append(f"| {Idx} | {Query_} | {Chunks} | {TopScore} |")
    else:
        Lines.append("_No retrieval operations this session._")
    Lines.append("")

    Lines.append("---")
    Lines.append("_Report generated by Multi-Agent Intelligence Terminal_")

    return "\n".join(Lines)

def _buildJson(SessionId, TimestampStr, SessionNumber, QAPairs, SafetyDecisions, Retrievals, UserPreferences, SessionLog,) -> dict:
    return {
        "session_id": SessionId,
        "session_number": SessionNumber,
        "generated_at": TimestampStr,
        "summary": {
            "total_queries": len(QAPairs),
            "blocked_queries": len([D for D in SafetyDecisions
                                     if D.get("verdict") == "UNSAFE"]),
            "total_retrievals": len(Retrievals),
            "total_chunks": sum(R.get("chunks_found", 0) for R in Retrievals),
        },
        "user_preferences": UserPreferences,
        "qa_pairs": QAPairs,
        "safety_decisions": SafetyDecisions,
        "retrieval_log": Retrievals,
        "full_session_log": SessionLog,
    }

def _extractByEvent(SessionLog: list[dict], EventName: str) -> list[dict]:
    return [Entry for Entry in SessionLog if Entry.get("event") == EventName]

def _extractQAPairs(SessionLog: list[dict]) -> list[dict]:
    
    QAPairs = []
    RoutingEntries = [E for E in SessionLog if E.get("event") == "supervisor_routing"]
    WritingEntries = [E for E in SessionLog if E.get("event") == "writing"]
    SafeInputs = [E for E in SessionLog if E.get("event") == "safety_input"]
    SafeOutputs = [E for E in SessionLog if E.get("event") == "safety_output"]

    for Idx, Routing in enumerate(RoutingEntries):
        Query = Routing.get("query", "")

        Writing = WritingEntries[Idx] if Idx < len(WritingEntries) else {}

        SafeIn  = SafeInputs[Idx]  if Idx < len(SafeInputs)  else {}
        SafeOut = SafeOutputs[Idx] if Idx < len(SafeOutputs) else {}

        InputSafe  = SafeIn.get("verdict", "SAFE") == "SAFE"
        OutputSafe = SafeOut.get("verdict", "SAFE") == "SAFE"

        QAPairs.append({
            "query": Query,
            "answer": Writing.get("answer", ""),
            "cited_sources": Writing.get("cited_sources", []),
            "input_safe": InputSafe,
            "output_safe": OutputSafe,
            "revision": Writing.get("revision", 0),
        })

    return QAPairs