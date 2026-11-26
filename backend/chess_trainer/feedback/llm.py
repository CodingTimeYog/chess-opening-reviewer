"""Optional LLM layer: a local Ollama model rewrites the template feedback as prose.

Off by default. Set CHESS_TRAINER_LLM=ollama to turn it on (model from
CHESS_TRAINER_LLM_MODEL, default qwen2.5:3b; server from OLLAMA_HOST). The model is given
only the structured facts and the template text, and is told to add nothing to them. If
the server is unreachable or answers with nothing, the caller keeps the template text.
"""

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable

DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_HOST = "http://localhost:11434"
# The LLM does not need engine internals or ids; fewer fields leave less to garble.
FACT_KEYS = ("side", "played", "classification", "cp_loss", "best", "master", "repertoire")

SYSTEM = (
    "You are a chess opening coach. Rewrite the feedback below as two to four plain sentences "
    "for a club player, explaining the idea. Use only the information in the FACTS and the "
    "TEMPLATE. Keep every move exactly as written. Do not mention any move, evaluation, "
    "player, game, event or year that is not in the FACTS. Do not invent variations."
)


def _post(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class OllamaRewriter:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = 60.0,
        post: Callable[[str, dict, float], dict] = _post,
    ) -> None:
        self.model = model
        self.url = host.rstrip("/") + "/api/generate"
        self.timeout = timeout
        self._post = post

    def prompt(self, facts: dict, template_text: str) -> str:
        slim = {key: facts[key] for key in FACT_KEYS if key in facts}
        slim["cited_games"] = [
            {k: c[k] for k in ("white", "black", "event", "year", "result", "move_san") if k in c}
            for c in facts.get("citations", [])
        ]
        return f"FACTS:\n{json.dumps(slim, indent=1)}\n\nTEMPLATE:\n{template_text}\n\nREWRITE:"

    def rewrite(self, facts: dict, template_text: str) -> str | None:
        payload = {
            "model": self.model,
            "system": SYSTEM,
            "prompt": self.prompt(facts, template_text),
            "stream": False,
            "options": {"temperature": 0.2, "seed": 0},
        }
        try:
            answer = self._post(self.url, payload, self.timeout)
        except (OSError, urllib.error.URLError, ValueError):
            return None
        return (answer.get("response") or "").strip() or None


def rewriter_from_env() -> OllamaRewriter | None:
    """The feature flag: None unless CHESS_TRAINER_LLM=ollama."""
    if os.environ.get("CHESS_TRAINER_LLM", "").lower() != "ollama":
        return None
    return OllamaRewriter(
        model=os.environ.get("CHESS_TRAINER_LLM_MODEL", DEFAULT_MODEL),
        host=os.environ.get("OLLAMA_HOST", DEFAULT_HOST),
    )
