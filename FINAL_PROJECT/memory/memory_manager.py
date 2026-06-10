import uuid
from datetime import datetime, timezone

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

Checkpointer = MemorySaver()       
LongTermStore = InMemoryStore() 

def newSessionId() -> str:
    DatePart = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ShortId  = str(uuid.uuid4())[:8]
    return f"session_{DatePart}_{ShortId}"

def buildThreadConfig(SessionId: str) -> dict:
    return {"configurable": {"thread_id": SessionId}}

_PREFS_NAMESPACE = ("user_prefs", "default_user")
_HISTORY_NAMESPACE = ("query_history", "default_user")

def saveUserPreference(Key: str, Value: str) -> None:
    LongTermStore.put(_PREFS_NAMESPACE, Key, {"value": Value})

def getUserPreference(Key: str, Default: str = "") -> str:
    try:
        Item = LongTermStore.get(_PREFS_NAMESPACE, Key)
        if Item is not None:
            return Item.value.get("value", Default)
    except Exception:
        pass
    return Default

def getAllPreferences() -> dict:
    try:
        Items = LongTermStore.search(_PREFS_NAMESPACE)
        return {Item.key: Item.value.get("value", "") for Item in Items}
    except Exception:
        return {}

def appendQueryHistory(SessionId: str, Query: str, Outcome: str) -> None:
    RecordKey = f"{SessionId}_{datetime.now(timezone.utc).strftime('%H%M%S%f')}"
    LongTermStore.put(
        _HISTORY_NAMESPACE,
        RecordKey,
        {
            "session": SessionId,
            "query": Query,
            "outcome": Outcome,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

def getQueryHistory(Limit: int = 20) -> list[dict]:
    try:
        Items = LongTermStore.search(_HISTORY_NAMESPACE)
        Records = [Item.value for Item in Items]
        Records.sort(key=lambda R: R.get("timestamp", ""), reverse=True)
        return Records[:Limit]
    except Exception:
        return []

def incrementSessionCount() -> int:
    Current = int(getUserPreference("session_count", "0"))
    New = Current + 1
    saveUserPreference("session_count", str(New))
    return New