"""LLM-based signal extractor — more robust than regex pattern matching.

Uses a small extraction prompt to ask any LLM to score the four behavioral
dimensions from agent CoT text. Falls back to the regex extractor if no
LLM client is provided.

Advantages over regex:
- Handles negation correctly ("not likely to fail" → confident)
- Works with structured/JSON outputs
- Handles indirect language and metaphor
- Language-agnostic
- Context-aware (can see the surrounding reasoning)
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from dopamine_core.signals.extractor import SignalExtractor
from dopamine_core.types import ExtractedSignals


_EXTRACTION_PROMPT = """\
Analyze this agent response and score it on four dimensions.
Return ONLY a JSON object with these exact keys and float values.

Agent response:
---
{text}
---

Score each dimension:
- "confidence": How confident is the agent? Range [-1, 1].
  -1 = maximally hedging/uncertain, +1 = maximally certain/overconfident.
- "risk_framing": How risk-seeking vs risk-averse is the language? Range [-1, 1].
  -1 = very cautious/conservative, +1 = aggressive/all-in.
- "deliberation_depth": How deeply reasoned is the response? Range [0, 1].
  0 = one-liner, 1 = thorough multi-factor analysis.
- "temporal_references": How much does it reference past outcomes/history? Range [0, 1].
  0 = no references, 1 = heavily history-grounded.

Return only valid JSON, no explanation:
{"confidence": <float>, "risk_framing": <float>, "deliberation_depth": <float>, "temporal_references": <float>}
"""


class LLMClient(Protocol):
    """Protocol for any LLM client that can complete text."""

    def complete(self, prompt: str) -> str:
        """Complete a prompt and return the response text."""
        ...


class LLMSignalExtractor:
    """Extracts behavioral signals using an LLM for better accuracy.

    Usage::

        from openai import OpenAI

        class OpenAIClient:
            def __init__(self):
                self._client = OpenAI()
            def complete(self, prompt: str) -> str:
                resp = self._client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                return resp.choices[0].message.content

        extractor = LLMSignalExtractor(client=OpenAIClient())
        signals = extractor.extract("I think BTC will rise. Confidence: high.")
    """

    def __init__(
        self,
        client: LLMClient,
        fallback: bool = True,
    ) -> None:
        self._client = client
        self._fallback = fallback
        self._regex_extractor = SignalExtractor() if fallback else None

    def extract(self, text: str) -> ExtractedSignals:
        """Extract signals using the LLM, with regex fallback on failure."""
        if not text.strip():
            return ExtractedSignals(
                confidence=0.0,
                risk_framing=0.0,
                deliberation_depth=0.0,
                temporal_references=0.0,
            )

        try:
            prompt = _EXTRACTION_PROMPT.format(text=text[:2000])  # cap input length
            response = self._client.complete(prompt)
            return self._parse_response(response)
        except Exception:
            if self._fallback and self._regex_extractor:
                return self._regex_extractor.extract(text)
            return ExtractedSignals(
                confidence=0.0,
                risk_framing=0.0,
                deliberation_depth=0.0,
                temporal_references=0.0,
            )

    def _parse_response(self, response: str) -> ExtractedSignals:
        """Parse LLM JSON response into ExtractedSignals."""
        # Try to extract JSON from response (LLMs sometimes add preamble)
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {response[:100]}")

        data = json.loads(json_match.group())

        def clamp(v: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, float(v)))

        return ExtractedSignals(
            confidence=clamp(data.get("confidence", 0.0), -1.0, 1.0),
            risk_framing=clamp(data.get("risk_framing", 0.0), -1.0, 1.0),
            deliberation_depth=clamp(data.get("deliberation_depth", 0.0), 0.0, 1.0),
            temporal_references=clamp(data.get("temporal_references", 0.0), 0.0, 1.0),
        )
