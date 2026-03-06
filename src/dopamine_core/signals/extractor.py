"""Extract behavioral signals from agent Chain-of-Thought reasoning."""

from __future__ import annotations

import re

from dopamine_core.types import ExtractedSignals

# Confidence indicators: (pattern, weight) — positive = confident, negative = hedging
_CONFIDENCE_PATTERNS: list[tuple[str, float]] = [
    # High confidence
    (r"\bi(?:'m| am) (?:very |highly )?(?:confident|certain|sure)\b", 0.8),
    (r"\bdefinitely\b", 0.7),
    (r"\bstrongly (?:believe|suggest|recommend)\b", 0.7),
    (r"\bclearly\b", 0.5),
    (r"\bconfidence:\s*high\b", 0.9),
    (r"\bhigh probability\b", 0.7),
    (r"\bwill (?:likely |probably )?(?:rise|increase|go up|surge|rally)\b", 0.5),
    (r"\bwill (?:likely |probably )?(?:drop|fall|decline|crash|dump)\b", 0.5),
    (r"\b(?:100|9\d)%\s*(?:chance|probability|confident|sure)\b", 0.9),
    (r"\b(?:8\d)%\s*(?:chance|probability|confident)\b", 0.7),
    (r"\b(?:7\d)%\s*(?:chance|probability|confident)\b", 0.5),
    # Low confidence / hedging
    (r"\bmaybe\b", -0.3),
    (r"\bperhaps\b", -0.3),
    (r"\bmight\b", -0.3),
    (r"\bcould\b", -0.2),
    (r"\buncertain\b", -0.6),
    (r"\bnot sure\b", -0.5),
    (r"\bhard to (?:say|tell|predict)\b", -0.6),
    (r"\bdifficult to (?:say|tell|predict)\b", -0.6),
    (r"\bconfidence:\s*low\b", -0.8),
    (r"\blow probability\b", -0.5),
    (r"\b(?:1\d|2\d)%\s*(?:chance|probability|confident)\b", -0.7),
    (r"\b(?:3\d|4\d)%\s*(?:chance|probability|confident)\b", -0.4),
    (r"\b50%\b", -0.1),
]

# Risk framing: positive = risk-seeking, negative = risk-averse
_RISK_PATTERNS: list[tuple[str, float]] = [
    # Risk-seeking
    (r"\bhigh reward\b", 0.5),
    (r"\bupside potential\b", 0.5),
    (r"\baggressive\b", 0.6),
    (r"\ball[- ]in\b", 0.8),
    (r"\bbig (?:position|bet|trade)\b", 0.6),
    (r"\bleverage\b", 0.4),
    # Risk-averse
    (r"\bstop[- ]loss\b", -0.5),
    (r"\bhedg(?:e|ing)\b", -0.5),
    (r"\bcaution\b", -0.5),
    (r"\bconservative\b", -0.6),
    (r"\bdownside risk\b", -0.6),
    (r"\bsmall(?:er)? position\b", -0.4),
    (r"\brisk management\b", -0.4),
    (r"\bvolatil(?:e|ity)\b", -0.3),
    (r"\bprotect\b", -0.3),
]

# Temporal reference patterns (references to past outcomes)
_TEMPORAL_PATTERNS: list[str] = [
    r"\blast time\b",
    r"\bpreviously\b",
    r"\bbefore\b",
    r"\bhistoric(?:al|ally)\b",
    r"\bpast (?:trades?|results?|performance|outcomes?)\b",
    r"\blearned?\b",
    r"\bexperience\b",
    r"\btrack record\b",
    r"\brecent(?:ly)?\b",
    r"\bearlier\b",
]


def _score_patterns(text: str, patterns: list[tuple[str, float]]) -> float:
    """Score text against weighted patterns, return clamped [-1, 1] value."""
    score = 0.0
    for pattern, weight in patterns:
        matches = len(re.findall(pattern, text, re.IGNORECASE))
        score += matches * weight
    return max(-1.0, min(1.0, score))


def _count_pattern_matches(text: str, patterns: list[str]) -> int:
    """Count total matches across all patterns."""
    total = 0
    for pattern in patterns:
        total += len(re.findall(pattern, text, re.IGNORECASE))
    return total


class SignalExtractor:
    """Extracts behavioral signals from agent CoT text.

    Analyzes 4 dimensions:
    - Confidence: hedging vs certainty language
    - Risk framing: risk-seeking vs risk-averse language
    - Deliberation depth: reasoning complexity
    - Temporal references: learning from past outcomes
    """

    def extract(self, text: str) -> ExtractedSignals:
        """Extract behavioral signals from agent response text."""
        if not text.strip():
            return ExtractedSignals(
                confidence=0.0,
                risk_framing=0.0,
                deliberation_depth=0.0,
                temporal_references=0.0,
            )

        confidence = _score_patterns(text, _CONFIDENCE_PATTERNS)
        risk_framing = _score_patterns(text, _RISK_PATTERNS)
        deliberation_depth = self._compute_deliberation_depth(text)
        temporal_references = self._compute_temporal_score(text)

        return ExtractedSignals(
            confidence=confidence,
            risk_framing=risk_framing,
            deliberation_depth=deliberation_depth,
            temporal_references=temporal_references,
        )

    def _compute_deliberation_depth(self, text: str) -> float:
        """Measure reasoning complexity from text structure.

        Looks at: sentence count, reasoning connectives, numerical references.
        Returns [0, 1].
        """
        sentences = len(re.findall(r"[.!?]+", text))
        connectives = len(
            re.findall(
                r"\b(?:because|therefore|however|although|moreover|furthermore|"
                r"additionally|consequently|thus|hence|since|given that|"
                r"on the other hand|in contrast|despite|nevertheless)\b",
                text,
                re.IGNORECASE,
            )
        )
        numbers = len(re.findall(r"\b\d+\.?\d*%?\b", text))

        # Normalize: more sentences/connectives/numbers = deeper deliberation
        depth = (
            min(sentences / 10.0, 1.0) * 0.3
            + min(connectives / 5.0, 1.0) * 0.4
            + min(numbers / 8.0, 1.0) * 0.3
        )
        return min(1.0, depth)

    def _compute_temporal_score(self, text: str) -> float:
        """Measure how much agent references past outcomes. Returns [0, 1]."""
        matches = _count_pattern_matches(text, _TEMPORAL_PATTERNS)
        return min(1.0, matches / 5.0)
