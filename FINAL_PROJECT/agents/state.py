import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):

    Messages: Annotated[list[BaseMessage], operator.add]
    UserQuery: str
    NextAgent: str
    InputSafe: bool    
    OutputSafe: bool    
    SafetyReason: str     
    SafetyDecisions: Annotated[list, operator.add] 

    RetrievedChunks: list    
    RetrievalScores: list    

    AnalystSummary: str      

    FinalAnswer: str
    CitedSources: list       

    
    RevisionCount: int
    MaxRevisions: int     
    
    SessionLog: Annotated[list, operator.add]

    McpToolResult: str