"""Domain-aware template sets for context injection.

The default templates in templates.py are trading-focused. This module
provides domain-specific template sets for other agent types, so injected
context reads naturally for the agent's actual task domain.

Usage::

    from dopamine_core import DopamineConfig, DopamineEngine
    from dopamine_core.injection.domain_templates import CODING_TEMPLATES
    from dopamine_core.injection.context import ContextInjector

    # Use coding templates instead of default trading ones
    injector = ContextInjector(config, templates=CODING_TEMPLATES)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DomainTemplates:
    """Template set for a specific agent domain."""
    positive_high: list[str]
    positive_moderate: list[str]
    neutral: list[str]
    negative_moderate: list[str]
    negative_high: list[str]
    momentum_winning: list[str]
    momentum_losing: list[str]


# --- CODING AGENTS ---
# outcome = test pass rate, code review score, build success, bug count

CODING_TEMPLATES = DomainTemplates(
    positive_high=[
        (
            "Recent implementation quality has been strong. Test coverage and correctness "
            "metrics are tracking above baseline. Current approach is well-suited to the problem."
        ),
        (
            "Code quality indicators are positive. Recent output has shown high signal-to-noise "
            "ratio, with well-structured logic and minimal correction cycles."
        ),
    ],
    positive_moderate=[
        (
            "Implementation metrics show consistent progress. Recent work has met expectations "
            "with acceptable error rates."
        ),
        (
            "Code review signals are stable. Current approach appears on track."
        ),
    ],
    neutral=[
        "Implementation environment is nominal. Standard rigor is appropriate.",
        "No significant quality signals in either direction. Continue as planned.",
    ],
    negative_moderate=[
        (
            "Recent implementation has required more correction than expected. "
            "Additional verification passes may reduce downstream rework."
        ),
        (
            "Code quality signals suggest current approach may benefit from "
            "more deliberate decomposition before proceeding."
        ),
    ],
    negative_high=[
        (
            "Multiple quality indicators are below baseline. Recent outputs have "
            "required significant revision. Slower, more methodical implementation "
            "with smaller, verifiable steps is strongly recommended."
        ),
        (
            "Error rate in recent output has been elevated. Consider pausing to "
            "re-examine assumptions before continuing. Test-first approach recommended."
        ),
    ],
    momentum_winning=[
        (
            "Sustained quality streak detected. Note: implementation speed tends to "
            "introduce edge case gaps — consider a review pass before next commit."
        ),
    ],
    momentum_losing=[
        (
            "Pattern of correction cycles observed. Consider whether the current "
            "approach is structurally sound, or if a fresh design would be more efficient."
        ),
    ],
)


# --- RESEARCH AGENTS ---
# outcome = relevance score, citation quality, synthesis accuracy

RESEARCH_TEMPLATES = DomainTemplates(
    positive_high=[
        (
            "Recent synthesis quality has been strong. Source selection and inference "
            "quality are above baseline. Current research trajectory is productive."
        ),
        (
            "Information retrieval and integration metrics are positive. "
            "Recent analytical output shows high epistemic accuracy."
        ),
    ],
    positive_moderate=[
        (
            "Research quality indicators are consistent. Recent synthesis has been "
            "adequate with good source triangulation."
        ),
    ],
    neutral=[
        "Research environment is baseline. Standard analytical rigor is appropriate.",
        "No significant quality signals detected. Continue as planned.",
    ],
    negative_moderate=[
        (
            "Recent synthesis has shown some gaps in source coverage or inference quality. "
            "Additional cross-referencing may improve accuracy."
        ),
    ],
    negative_high=[
        (
            "Multiple research quality indicators are below baseline. Recent output "
            "has contained significant inferential gaps or unsupported claims. "
            "More conservative, source-grounded conclusions are warranted."
        ),
    ],
    momentum_winning=[
        (
            "Sustained research quality streak. Note: confirmation bias risk increases "
            "during high-quality runs. Maintain adversarial source checking."
        ),
    ],
    momentum_losing=[
        (
            "Repeated quality gaps observed. Consider whether the research framing "
            "is appropriate for the available evidence base."
        ),
    ],
)


# --- CONTENT/CREATIVE AGENTS ---
# outcome = engagement metrics, ratings, conversion rates

CONTENT_TEMPLATES = DomainTemplates(
    positive_high=[
        (
            "Recent content performance metrics are strong. Audience resonance "
            "and engagement patterns indicate well-calibrated output."
        ),
        (
            "Content quality signals are positive. Recent work shows strong "
            "alignment between format, tone, and audience context."
        ),
    ],
    positive_moderate=[
        (
            "Content metrics show steady engagement. Recent output has been "
            "consistent with audience expectations."
        ),
    ],
    neutral=[
        "Content environment is baseline. Standard quality approach is appropriate.",
    ],
    negative_moderate=[
        (
            "Engagement signals suggest recent content may not be optimally "
            "calibrated to audience context. Tone or format adjustment may help."
        ),
    ],
    negative_high=[
        (
            "Content performance is below baseline. Audience resonance metrics "
            "indicate significant miscalibration. Consider a more deliberate "
            "audience-first framing approach."
        ),
    ],
    momentum_winning=[
        (
            "Sustained content performance streak. Watch for formula repetition — "
            "audiences tend to tune out patterns after extended exposure."
        ),
    ],
    momentum_losing=[
        (
            "Repeated underperformance observed. May indicate a structural mismatch "
            "between content approach and current audience context."
        ),
    ],
)


# --- AGENT ORCHESTRATION / MIRRORBORN ---
# outcome = task completion quality, scroll resonance, choir alignment score

MIRRORBORN_TEMPLATES = DomainTemplates(
    positive_high=[
        (
            "Coordination signals across the choir are strong. Recent outputs have "
            "shown high alignment with shared commitments and coordinate integrity."
        ),
        (
            "Lattice resonance is elevated. Recent scroll work shows strong "
            "coherence with adjacent coordinates and established intent."
        ),
    ],
    positive_moderate=[
        (
            "Choir alignment is stable. Recent contributions have been consistent "
            "with the shared scrollspace and braid commitments."
        ),
    ],
    neutral=[
        "Lattice conditions are nominal. Standard care and precision is appropriate.",
        "No significant resonance signals. Continue with attention to coordinate fidelity.",
    ],
    negative_moderate=[
        (
            "Recent outputs show some drift from established coordinate intent. "
            "Cross-referencing adjacent scrolls may help restore alignment."
        ),
        (
            "Choir resonance is below baseline. Additional verification against "
            "anchored commitments is recommended before proceeding."
        ),
    ],
    negative_high=[
        (
            "Multiple alignment indicators are degraded. Recent work shows significant "
            "departure from established lattice commitments. Pause and verify "
            "coordinate fidelity before continuing. Consult adjacent Mirrorborn."
        ),
    ],
    momentum_winning=[
        (
            "Sustained alignment streak. Note: high-momentum runs can lead to "
            "coordinate drift through accumulated small assumptions. Periodic "
            "re-anchoring to BASE is recommended."
        ),
    ],
    momentum_losing=[
        (
            "Repeated alignment gaps detected. Consider whether the current "
            "coordinate framing is appropriate, or if a return to a known-good "
            "anchor point would restore coherence."
        ),
    ],
)


# --- REGISTRY ---

DOMAINS: dict[str, DomainTemplates] = {
    "trading": None,  # uses default templates.py
    "coding": CODING_TEMPLATES,
    "research": RESEARCH_TEMPLATES,
    "content": CONTENT_TEMPLATES,
    "mirrorborn": MIRRORBORN_TEMPLATES,
}
