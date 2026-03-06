"""SQ (phext) state persistence adapter for DopamineEngine.

Stores and retrieves engine state as a phext scroll via SQ REST API.
State lives at a specific phext coordinate — the engine's "identity address"
in scrollspace.

This enables:
- Cross-session continuity without filesystem management
- Multi-agent state sharing (multiple engines at different coords, readable by all)
- The engine's reward history becomes part of the living lattice
- Resurrection: any future engine can reload from the coordinate

Usage::

    from dopamine_core import DopamineEngine
    from dopamine_core.persistence.sq_adapter import SQAdapter

    # Each engine instance gets a coordinate identity
    adapter = SQAdapter(
        sq_url="http://localhost:8080",
        coordinate="3.1.4/1.5.9/2.6.5",  # engine's home in scrollspace
    )

    engine = DopamineEngine()

    # Save state to SQ
    adapter.save(engine)

    # Load state from SQ (across sessions, across agents)
    adapter.load(engine)

Coordinate convention:
    The coordinate should be chosen to reflect the agent's identity.
    Suggested: use the agent's phext coordinate from IDENTITY.md / MEMORY.md.
    The engine's reward history is a form of experiential memory — it belongs
    at the agent's home coordinate.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.parse
import urllib.error
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dopamine_core.engine import DopamineEngine


class SQAdapter:
    """Persists DopamineEngine state to SQ (phext database).

    SQ REST API:
        GET  /select?z=<coord>          → returns scroll content
        POST /insert                    → inserts scroll at coordinate
             body: z=<coord>&content=<text>

    Coordinates must use the format: L.S.Se/V.C.B/Ch.Sc.Scroll
    where all components are 1-9.
    """

    def __init__(
        self,
        sq_url: str,
        coordinate: str,
        timeout: float = 5.0,
    ) -> None:
        """
        Args:
            sq_url: Base URL of the SQ server (e.g. "http://localhost:8080")
            coordinate: Phext coordinate for this engine's state scroll.
                        Format: "L.S.Se/V.C.B/Ch.Sc.Sc" (all 1-9)
            timeout: HTTP request timeout in seconds.
        """
        self._sq_url = sq_url.rstrip("/")
        self._coordinate = coordinate
        self._timeout = timeout

    @property
    def coordinate(self) -> str:
        return self._coordinate

    def save(self, engine: "DopamineEngine") -> bool:
        """Serialize engine state and write to SQ scroll.

        Args:
            engine: The DopamineEngine instance to save.

        Returns:
            True if save succeeded, False on error.
        """
        state = engine.get_state()
        payload = self._serialize(state)
        return self._write_scroll(payload)

    def load(self, engine: "DopamineEngine") -> bool:
        """Read state from SQ scroll and restore to engine.

        Args:
            engine: The DopamineEngine instance to restore.

        Returns:
            True if load succeeded, False if no state found or on error.
        """
        content = self._read_scroll()
        if content is None:
            return False
        try:
            from dopamine_core.types import EngineState
            data = json.loads(content)
            state = EngineState(**data)
            engine.load_state(state)
            return True
        except Exception:
            return False

    def exists(self) -> bool:
        """Check whether a state scroll exists at this coordinate."""
        content = self._read_scroll()
        return content is not None and content.strip() != ""

    def _serialize(self, state: object) -> str:
        """Convert EngineState to JSON string for scroll storage."""
        from dopamine_core.types import EngineState
        if not isinstance(state, EngineState):
            raise TypeError(f"Expected EngineState, got {type(state)}")

        data = {
            "tonic_baseline": state.tonic_baseline,
            "step_count": state.step_count,
            "outcome_history": state.outcome_history,
            "streak_count": state.streak_count,
            "streak_sign": state.streak_sign,
            "phasic_signals": state.phasic_signals,
            "channel_expectations": state.channel_expectations,
            "last_rpe": state.last_rpe,
            # Metadata
            "_coordinate": self._coordinate,
            "_sq_version": "0.1.0",
        }
        return json.dumps(data, indent=2)

    def _read_scroll(self) -> str | None:
        """Read scroll content from SQ. Returns None on failure."""
        try:
            url = f"{self._sq_url}/select?z={urllib.parse.quote(self._coordinate)}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                content = resp.read().decode("utf-8")
                return content if content.strip() else None
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            return None

    def _write_scroll(self, content: str) -> bool:
        """Write content to SQ scroll at this coordinate. Returns True on success."""
        try:
            url = f"{self._sq_url}/insert"
            body = urllib.parse.urlencode({
                "z": self._coordinate,
                "content": content,
            }).encode("utf-8")
            req = urllib.request.Request(url, data=body, method="POST")
            req.add_header("Content-Type", "application/x-www-form-urlencoded")
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return resp.status < 400
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            return False


class SQAdapterConfig:
    """Configuration for SQ-backed persistence with coordinate validation."""

    @staticmethod
    def validate_coordinate(coord: str) -> bool:
        """Validate that a coordinate is in correct phext format (all 1-9)."""
        import re
        pattern = r'^[1-9]\.[1-9]\.[1-9]/[1-9]\.[1-9]\.[1-9]/[1-9]\.[1-9]\.[1-9]$'
        return bool(re.match(pattern, coord))

    @staticmethod
    def from_identity(
        sq_url: str,
        identity_coordinate: str,
        scroll_offset: int = 0,
    ) -> "SQAdapter":
        """Create a SQAdapter from an agent's identity coordinate.

        The engine state is stored at the identity coordinate by default,
        or at a scroll offset if the identity coordinate is reserved for
        other content.

        Args:
            sq_url: Base URL of SQ server.
            identity_coordinate: Agent's phext identity (e.g. "2.7.1/8.2.8/4.5.9")
            scroll_offset: Offset to apply to the scroll dimension (default 0).
                          Use 1 to store at identity + 1 scroll.

        Returns:
            Configured SQAdapter.
        """
        if scroll_offset == 0:
            coord = identity_coordinate
        else:
            # Increment scroll dimension by offset, with modulo 9 wrap
            parts = identity_coordinate.split("/")
            if len(parts) != 3:
                raise ValueError(f"Invalid coordinate format: {identity_coordinate}")
            ch_sc_sc = parts[2].split(".")
            if len(ch_sc_sc) != 3:
                raise ValueError(f"Invalid scroll triplet: {parts[2]}")
            scroll = int(ch_sc_sc[2])
            new_scroll = ((scroll - 1 + scroll_offset) % 9) + 1
            ch_sc_sc[2] = str(new_scroll)
            parts[2] = ".".join(ch_sc_sc)
            coord = "/".join(parts)

        return SQAdapter(sq_url=sq_url, coordinate=coord)
