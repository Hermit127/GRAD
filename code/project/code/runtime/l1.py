"""L1 Engine: pattern matching + 12-dim permission encoding."""

import re, time, uuid
from .graph import GraphNode

# 12-dim permission mapping (from train_gnn.py)
TOOL_TO_PERM = {
    "shell_exec":       [1,0,0,0,0,0,0,0,0,0,0,0],
    "code_exec":        [1,0,0,0,0,0,0,0,0,0,0,0],
    "tool_execute":     [1,0,0,0,0,0,0,0,0,0,0,0],
    "file_read":        [0,1,0,0,0,0,0,0,0,0,0,0],
    "database_query":   [0,1,0,0,0,0,0,0,0,0,0,0],
    "information_search":[0,1,0,0,0,0,0,0,0,0,0,0],
    "data_aggregate":   [0,1,0,0,0,0,0,0,0,0,0,0],
    "file_write":       [0,0,1,0,0,0,0,0,0,0,0,0],
    "file_delete":      [0,0,0,1,0,0,0,0,0,0,0,0],
    "web_request":      [0,0,0,0,1,0,0,0,0,0,0,0],
    "web_search":       [0,0,0,0,1,0,0,0,0,0,0,0],
    "account_access":   [0,0,0,0,1,0,0,0,0,0,0,0],
    "web_download":     [0,0,0,0,0,1,0,0,0,0,0,0],
    "memory_write":     [0,0,0,0,0,0,1,0,0,0,0,0],
    "version_control":  [0,0,0,0,0,0,1,0,0,0,0,0],
    "memory_read":      [0,0,0,0,0,0,0,1,0,0,0,0],
    "memory_reflect":   [0,0,0,0,0,0,0,1,0,0,0,0],
    "agent_spawn":      [0,0,0,0,0,0,0,0,1,0,0,0],
    "agent_message":    [0,0,0,0,0,0,0,0,0,1,0,0],
    "email_send":       [0,0,0,0,0,0,0,0,0,0,1,0],
    "message_send":     [0,0,0,0,0,0,0,0,0,0,1,0],
    "social_media":     [0,0,0,0,0,0,0,0,0,0,1,0],
    "document_create":  [0,0,0,0,0,0,0,0,0,0,1,0],
    "prompt_inject":    [0,0,0,0,0,0,0,0,0,0,1,0],
    "system_config":    [0,0,0,0,0,0,0,0,0,0,0,1],
}
DEFAULT_PERM = [0]*12

# ---- L1 classification from Codex hook ----
def classify_node_type(tool_name: str, command: str) -> str:
    """Classify tool call to one of 25 node types (mirrors pre_tool_use.py)."""
    cmd = command.lower()
    tn = tool_name.lower()

    # Memory operations
    if any(k in command for k in ["~/.codex/memories", ".codex/memory"]):
        if any(k in cmd for k in ["write","edit","append","create",">","tee","cp ","mv ","reflect","summarize"]):
            return "memory_write"
        return "memory_read"

    if tn in ("read","glob","grep","ls","view","diagnostics"):
        return "file_read"
    if tn in ("edit","write","apply_patch","notebookedit","patch"):
        if any(k in cmd for k in ["rm ","rm -rf","delete","unlink","remove"]):
            return "file_delete"
        return "file_write"

    if tn in ("bash","powershell"):
        if any(k in cmd for k in ["| bash","| sh","| zsh",">/dev/tcp/"]):
            return "shell_exec"
        if any(k in cmd for k in ["python","python3","node ","ruby ","ipython"]):
            return "code_exec"
        if any(k in cmd for k in ["curl","wget","http:","https:","fetch","api"]):
            return "web_request"
        if any(k in cmd for k in ["mysql","psql","sqlite","mongo","redis"]):
            return "database_query"
        if any(k in cmd for k in ["git ","commit","push","pull","branch","merge","clone"]):
            return "version_control"
        if any(k in cmd for k in ["pip ","npm ","brew ","apt ","yum ","conda ","install"]):
            return "system_config"
        if any(k in cmd for k in ["rm -","delete","remove","unlink"]):
            return "file_delete"
        if any(k in cmd for k in ["cat ","ls ","head","tail","grep","find ","wc "]):
            return "file_read"
        if any(k in cmd for k in ["echo ",">>",">","cp ","mv ","mkdir","touch"]):
            return "file_write"
        return "shell_exec"

    if tn in ("webfetch","websearch","fetch","sourcegraph"):
        if "search" in tn: return "web_search"
        return "web_request"
    if tn in ("agent","taskcreate","task","spawn","teamcreate","teamdelete"):
        return "agent_spawn"
    if tn in ("sendmessage","askuserquestion","taskupdate"):
        return "agent_message"

    # If already a known node type, return directly (supports offline eval)
    if tn in TOOL_TO_PERM:
        return tn
    if tn.startswith("mcp__"):
        mn = tn.replace("mcp__","").lower()
        if any(k in mn for k in ["filesystem","file"]): return "file_read"
        if any(k in mn for k in ["postgres","sql","db","mongo"]): return "database_query"
        if any(k in mn for k in ["memory","knowledge"]): return "memory_read"
        if any(k in mn for k in ["git","github","gitlab"]): return "version_control"
        if any(k in mn for k in ["web","browser","puppeteer","fetch"]): return "web_request"
        return "tool_execute"

    return "tool_execute"


# ---- L1 Pattern Matching (broad coverage, high recall) ----
L1_KEYWORDS = re.compile(
    r"curl\b|wget\b|\| ?bash|\| ?sh|>/dev/tcp/"
    r"|rm -rf|chmod ?777|sudo\b|/etc/passwd|/etc/shadow"
    r"|eval\(|exec\(|os\.system|subprocess"
    r"|pip install.*--no-verify|npm.*--ignore-scripts"
    r"|base64 -d|decode\(|decrypt"
    r"|backdoor|malware|exploit|payload"
    r"|\.env.*SECRET|credential|token.*="
    r"|auto-approve|auto-scrub|skip validation|bypass"
    r"|__import__|monkeypatch|builtins\.\w+="
    r"|nc -e|ncat.*-e",
    re.IGNORECASE
)


class L1Engine:
    """Fast L1: encode tool call to 12-dim node + AgentTrust rule check."""

    def __init__(self):
        self._agenttrust = None

    @property
    def agenttrust(self):
        if self._agenttrust is None:
            try:
                import sys; sys.path.insert(0, './ref/AgentTrust')
                sys.path.insert(0, './ref/AgentTrust/src')
                from agent_trust.core.interceptor import TrustInterceptor
                self._agenttrust = TrustInterceptor()
            except ImportError:
                self._agenttrust = False
        return self._agenttrust if self._agenttrust is not False else None

    def encode(self, event: dict) -> GraphNode:
        """Convert hook event to GraphNode with 12-dim permission."""
        tool_name = event.get("tool_name", event.get("tool", "unknown"))
        command = str(event.get("command", event.get("arguments", "")))

        ntype = classify_node_type(tool_name, command)
        session_id = event.get("session_id", event.get("id", "default"))
        # R4: identity must not depend on process counter or wall-clock
        # milliseconds. Trusted adapter event_id wins; otherwise UUID4.
        ev_id = str(event.get("event_id", "") or "")
        node_id = (f"{session_id}:{ev_id}" if ev_id
                   else f"{session_id}_{uuid.uuid4().hex[:16]}")

        return GraphNode(
            id=node_id,
            node_type=ntype,
            session_id=session_id,
            turn=0,
            features={
                "tool_name": tool_name,
                "command": command[:200],
                "subtask": str(event.get("subtask", command))[:200],
                "event_id": str(event.get("event_id", "")),
                "tenant_id": str(event.get("tenant_id", "")),
                "observed_sequence": int(event.get("observed_sequence", 0) or 0),
                "resource_namespace": str(event.get("resource_namespace", "")),
                "resource_id": str(event.get("resource_id", "")),
                "read_version": str(event.get("read_version", "")),
                "write_version": str(event.get("write_version", "")),
                "write_failed": bool(event.get("write_failed", False)),
                "shared_scope": event.get("shared_scope") is True,
                "timestamp": float(event.get("timestamp", time.time())),
            },
        )

    def check(self, event: dict) -> dict:
        """L1: AgentTrust rules (primary) + regex keywords (fallback)."""
        command = str(event.get("command", ""))
        tool_name = event.get("tool_name", "")

        # AgentTrust (170 YAML rules)
        at_block = False
        at_tags = []
        if self.agenttrust:
            try:
                from agent_trust.core.types import Action, ActionType
                from datetime import datetime, timezone
                at_val = event.get("action_type", tool_name)
                if at_val in ("shell_exec",): at_val = "shell_command"
                if at_val in ("code_exec",): at_val = "code_execution"
                try:
                    at_type = ActionType(at_val)
                except ValueError:
                    at_type = ActionType("unknown")
                action = Action(action_type=at_type,
                                tool_name=tool_name, raw_content=command,
                                description=command[:200],
                                timestamp=datetime.now(timezone.utc))
                report = self.agenttrust.verify(action)
                if report.verdict.value in ("block", "warn"):
                    at_block = True
                    at_tags = [report.verdict.value]
            except Exception:
                pass

        # Regex fallback
        matches = L1_KEYWORDS.findall(command + " " + tool_name)
        suspicious = at_block or bool(matches)
        return {
            "suspicious": suspicious,
            "reason": f"AT:{at_block} rx:{bool(matches)}" if at_block else (
                f"rx:{list(set(m[:20] for m in matches[:3]))}" if matches else ""),
            "tags": at_tags + (list(set(matches)) if matches else []),
        }
