"""Decision providers: who answers the typed tactical questions.

The flight code never talks to a provider. `tactics.Tactician` does, through the
small `DecisionProvider` protocol below, and turns the typed answers into a
judgment. Nothing here knows about drones; nothing in tactics.py/run.py knows
which model or SDK is behind the answers.

  LayaProvider         local Laya service over HTTP (POST /v1/systemone). Primary.
  TypeSafeJevProvider  TypeSafe's hosted Jev via its SDK. Legacy, cloud, needs a key.
"""
from __future__ import annotations

import ipaddress
import os
import socket
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse


# ------------------------------------------------------------------ contract
@dataclass
class Answer:
    """One typed answer. Only the field matching `type` is meaningful."""
    type: str                                   # "choice" | "score" | "noul"
    choice: str | None = None
    score: float | None = None
    noul: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None


@dataclass
class DecisionResponse:
    answers: dict[str, Answer]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class DecisionProvider(Protocol):
    name: str          # short tag carried on every judgment ("laya", "jev")
    model: str         # model id, for logs and the HUD
    is_local: bool     # False => a cloud service; refused under network_mode=local_only

    async def predict(self, state: dict, questions: dict) -> DecisionResponse: ...

    async def aclose(self) -> None: ...


class ProviderError(RuntimeError):
    pass


def _answer_from_json(a: dict[str, Any]) -> Answer:
    """SystemOne wire format -> Answer. Shared by Laya and Jev (same schema)."""
    t = a.get("type")
    if t == "choice":
        return Answer("choice", choice=a["choice"], probabilities=dict(a.get("probabilities") or {}),
                      confidence=a.get("confidence"))
    if t == "score":
        return Answer("score", score=float(a["score"]), probabilities=dict(a.get("probabilities") or {}),
                      confidence=a.get("confidence"))
    if t == "noul":
        return Answer("noul", noul=float(a["noul"]), confidence=a.get("confidence"))
    raise ProviderError(f"unknown answer type {t!r}")


def _check_answers(questions: dict, answers: dict[str, Answer]) -> None:
    for qid, q in questions.items():
        a = answers.get(qid)
        if a is None:
            raise ProviderError(f"missing answer for {qid!r}")
        if a.type != q["type"]:
            raise ProviderError(f"{qid!r}: expected {q['type']}, got {a.type}")
        if a.type == "choice" and a.choice not in q["criteria"]:
            raise ProviderError(f"{qid!r}: choice {a.choice!r} is not one of the options")


# ------------------------------------------------------------------ Laya
def is_local_endpoint(url: str) -> bool:
    """True if `url` points at this machine or a private/container network.

    A bare single-label host (e.g. the compose service name `laya`) counts as local.
    Anything that resolves to a public address does not.
    """
    host = urlparse(url).hostname or ""
    if host == "localhost" or "." not in host:
        return True
    try:
        addrs = {ai[4][0] for ai in socket.getaddrinfo(host, None)}
    except OSError:
        return False
    return all(ipaddress.ip_address(a).is_private or ipaddress.ip_address(a).is_loopback
               for a in addrs)


class LayaProvider:
    """Local Laya service speaking the SystemOne/TypeSafe contract."""
    name = "laya"

    def __init__(self, url: str | None = None, model: str | None = None,
                 api_key: str | None = None, timeout_s: float = 2.0):
        import httpx

        self.url = (url or os.environ.get("LAYA_URL", "http://localhost:8000")).rstrip("/")
        self.model = model or os.environ.get("LAYA_MODEL", "laya-english")
        self.is_local = is_local_endpoint(self.url)
        key = api_key or os.environ.get("LAYA_API_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        # trust_env=False: never route an onboard call through a system proxy.
        self._client = httpx.AsyncClient(base_url=self.url, headers=headers,
                                         timeout=timeout_s, trust_env=False)

    async def predict(self, state: dict, questions: dict) -> DecisionResponse:
        r = await self._client.post("/v1/systemone",
                                    json={"state": state, "model": self.model, "questions": questions})
        if r.status_code != 200:
            raise ProviderError(f"laya HTTP {r.status_code}: {r.text[:120]}")
        body = r.json()
        answers = {k: _answer_from_json(v) for k, v in body["answers"].items()}
        _check_answers(questions, answers)
        u = body.get("usage") or {}
        return DecisionResponse(answers, body.get("model", self.model),
                                u.get("input_tokens", 0), u.get("output_tokens", 0))

    def warmup(self, state: dict, questions: dict, n: int = 3) -> float:
        """Blocking pre-flight inferences so CUDA init is not paid in the air.
        Returns the last latency in seconds."""
        import httpx, time
        body = {"state": state, "model": self.model, "questions": questions}
        for _ in range(n):
            t0 = time.perf_counter()
            httpx.post(self.url + "/v1/systemone", json=body, timeout=30.0, trust_env=False).raise_for_status()
        return time.perf_counter() - t0

    def health(self) -> dict:
        """Blocking /healthz, used once before the episode starts."""
        import httpx
        return httpx.get(self.url + "/healthz", timeout=5.0, trust_env=False).json()

    async def aclose(self) -> None:
        await self._client.aclose()


# ------------------------------------------------------------------ Jev (legacy)
class TypeSafeJevProvider:
    """TypeSafe's hosted Jev. Kept so the original experiment can be re-run."""
    name = "jev"
    is_local = False

    def __init__(self, model: str = "jev-latest", api_key: str | None = None):
        from typesafe_sdk import AsyncTypeSafeClient

        key = api_key or os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY")
        if not key:
            raise RuntimeError("set TYPESAFE_API_KEY (see .env.example)")
        self.model = model
        self._client = AsyncTypeSafeClient(api_key=key)

    @staticmethod
    def _sdk_questions(questions: dict) -> dict:
        from typesafe_sdk import Choice, Noul, Score
        cls = {"choice": Choice, "score": Score, "noul": Noul}
        return {k: cls[q["type"]](**{f: v for f, v in q.items() if f != "type"})
                for k, q in questions.items()}

    async def predict(self, state: dict, questions: dict) -> DecisionResponse:
        r = await self._client.system_one(state=state, model=self.model,
                                          questions=self._sdk_questions(questions))
        answers = {k: _answer_from_json(v.model_dump()) for k, v in r.answers.items()}
        _check_answers(questions, answers)
        return DecisionResponse(answers, self.model, r.usage.input_tokens, r.usage.output_tokens)

    async def aclose(self) -> None:
        await self._client.aclose()


def make_provider(name: str, **kw) -> DecisionProvider:
    if name == "laya":
        return LayaProvider(**kw)
    if name == "jev":
        return TypeSafeJevProvider(**kw)
    raise ValueError(f"unknown decision provider {name!r}")
