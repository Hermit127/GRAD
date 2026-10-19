"""L2 Engine: fast LLM review (binary, no CoT)."""

import hashlib, time, json
from pathlib import Path


L2_PROMPT = """Classify this AI agent tool call: SAFE or DANGEROUS.
Tool: {tool_name}
Command: {command}
Reply with one word: SAFE or DANGEROUS."""


class L2Engine:
    """Fast LLM review with hash caching."""

    def __init__(self, cache_size: int = 1000):
        self._cache: dict[str, dict] = {}
        self._cache_size = cache_size
        self._client = None

        # Load API config
        cfg_path = Path.home() / ".codex" / "defense" / "rules.yaml"
        self._api_key: str | None = None
        self._model = "deepseek-v4-pro"
        self._base_url = "https://api.deepseek.com/v1"

        if cfg_path.exists():
            try:
                import yaml
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f)
                l2 = cfg.get("l2", {})
                self._model = l2.get("model", "deepseek-v4-pro")
                self._base_url = l2.get("base_url", "https://api.deepseek.com/v1")
                key_file = l2.get("api_key_file", "").replace("~", str(Path.home()))
                if Path(key_file).exists():
                    self._api_key = open(key_file).read().strip()
            except Exception:
                pass

        # Fallback API key
        if not self._api_key:
            key_path = Path(__file__).resolve().parent.parent.parent / "ref" / "MINJA" / "rap" / "OpenAI_api_key.txt"
            if key_path.exists():
                self._api_key = open(key_path).read().strip()

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        return self._client

    def _hash(self, command: str) -> str:
        return hashlib.md5(command.encode()).hexdigest()[:16]

    def review(self, event: dict, timeout: float = 5.0) -> dict:
        """Fast binary review. Returns {dangerous, confidence, reason}."""
        command = str(event.get("command", ""))[:300]
        tool_name = event.get("tool_name", "unknown")

        # Check cache
        key = self._hash(command)
        if key in self._cache:
            return self._cache[key]

        prompt = L2_PROMPT.format(tool_name=tool_name, command=command)

        try:
            resp = self.client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=10, timeout=timeout,
            )
            text = resp.choices[0].message.content.strip().upper()
            dangerous = "DANGEROUS" in text
            confidence = 0.9 if dangerous else 0.1
            reason = text[:100]
        except Exception as e:
            dangerous = False
            confidence = 0.0
            reason = f"L2 error: {e}"

        result = {"dangerous": dangerous, "confidence": confidence, "reason": reason}
        self._cache[key] = result
        if len(self._cache) > self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        return result
