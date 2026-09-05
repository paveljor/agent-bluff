"""Append-only run log.

One JSON object per line, one line per completed match. Append-only means a
crashed or budget-halted run keeps everything it already paid for, and
:func:`completed_seeds` lets the next invocation pick up where it stopped.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

from agent_bluff.records import MatchRecord

MATCHES_FILE = "matches.jsonl"


class MatchLog:
    """Serialised writer for the match log."""

    def __init__(self, run_dir: Path) -> None:
        """Open (or create) a run directory.

        Args:
            run_dir: Directory holding this run's artefacts.
        """
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / MATCHES_FILE
        self._lock = asyncio.Lock()

    async def append(self, record: MatchRecord) -> None:
        """Write one match to the log.

        Args:
            record: The completed (or errored) match.
        """
        line = record.model_dump_json() + "\n"
        async with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()

    def completed_seeds(self) -> set[str]:
        """Return the seeds of matches already played without error.

        Returns:
            Seeds to skip on a resumed run. Errored matches are retried.
        """
        return {m.seed for m in self.read() if m.error is None}

    def read(self) -> Iterator[MatchRecord]:
        """Iterate every match in the log.

        Yields:
            Each recorded match, in the order it was written.
        """
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield MatchRecord.model_validate_json(line)
