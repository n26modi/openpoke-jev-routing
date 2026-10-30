"""Agent roster with per-agent metadata used by the retrieval pipeline."""

from __future__ import annotations

import json
import fcntl
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ...logging_config import logger


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AgentRecord:
    """Metadata we track per execution agent for routing and retrieval."""

    name: str
    description: str = ""
    created_at: str = field(default_factory=_utcnow_iso)
    last_used_at: str = field(default_factory=_utcnow_iso)
    use_count: int = 0
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AgentRecord":
        return cls(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            created_at=str(data.get("created_at") or _utcnow_iso()),
            last_used_at=str(data.get("last_used_at") or _utcnow_iso()),
            use_count=int(data.get("use_count", 0)),
            embedding=list(data["embedding"]) if data.get("embedding") else None,
        )


class AgentRoster:
    """Persistent roster of execution agents with routing metadata."""

    def __init__(self, roster_path: Path):
        self._roster_path = roster_path
        self._records: dict[str, AgentRecord] = {}
        self.load()

    # ---- persistence ----

    def load(self) -> None:
        """Load records from roster.json. Migrates legacy array-of-strings on the fly."""
        if not self._roster_path.exists():
            self._records = {}
            self.save()
            return

        try:
            with open(self._roster_path, "r") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f"Failed to load roster.json: {exc}")
            self._records = {}
            return

        if not isinstance(data, list):
            self._records = {}
            return

        records: dict[str, AgentRecord] = {}
        migrated = 0
        for entry in data:
            if isinstance(entry, str):
                record = AgentRecord(name=entry)
                migrated += 1
            elif isinstance(entry, dict) and entry.get("name"):
                try:
                    record = AgentRecord.from_dict(entry)
                except Exception as exc:
                    logger.warning(f"Skipping malformed roster entry: {exc}")
                    continue
            else:
                continue
            records[record.name] = record

        self._records = records
        if migrated:
            logger.info(f"Migrated {migrated} legacy roster entries")
            self.save()

    def save(self) -> None:
        """Persist records with file locking and exponential backoff on contention."""
        max_retries = 5
        retry_delay = 0.1

        for attempt in range(max_retries):
            try:
                self._roster_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._roster_path, "w") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    try:
                        payload = [r.to_dict() for r in self._records.values()]
                        json.dump(payload, f, indent=2)
                        return
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except BlockingIOError:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.warning("Failed to acquire lock on roster.json after retries")
            except Exception as exc:
                logger.warning(f"Failed to save roster.json: {exc}")
                break

    # ---- mutation ----

    def add_agent(self, name: str, description: str = "") -> AgentRecord:
        """Create or update an agent entry. If it exists, description is only set when non-empty."""
        existing = self._records.get(name)
        if existing is None:
            record = AgentRecord(name=name, description=description)
            self._records[name] = record
            self.save()
            return record

        if description and not existing.description:
            existing.description = description
            self.save()
        return existing

    def record_use(self, name: str) -> None:
        """Bump last_used_at + use_count for hot-cache ranking."""
        record = self._records.get(name)
        if record is None:
            return
        record.last_used_at = _utcnow_iso()
        record.use_count += 1
        self.save()

    def set_embedding(self, name: str, embedding: list[float]) -> None:
        """Cache the embedding vector so we don't recompute on every retrieval."""
        record = self._records.get(name)
        if record is None:
            return
        record.embedding = list(embedding)
        self.save()

    def set_description(self, name: str, description: str) -> None:
        """Overwrite the description and invalidate the cached embedding."""
        record = self._records.get(name)
        if record is None:
            return
        record.description = description
        record.embedding = None
        self.save()

    def clear(self) -> None:
        self._records = {}
        try:
            if self._roster_path.exists():
                self._roster_path.unlink()
            logger.info("Cleared agent roster")
        except Exception as exc:
            logger.warning(f"Failed to clear roster.json: {exc}")

    # ---- queries ----

    def get_agents(self) -> list[str]:
        """Legacy accessor: just agent names, used by UI and anything that didn't migrate."""
        return [r.name for r in self._records.values()]

    def all_records(self) -> list[AgentRecord]:
        return list(self._records.values())

    def get_by_name(self, name: str) -> Optional[AgentRecord]:
        return self._records.get(name)

    def get_hot_cache(self, n: int = 5) -> list[AgentRecord]:
        """Most-recently-used first. Ties broken by use_count then name."""
        return sorted(
            self._records.values(),
            key=lambda r: (r.last_used_at, r.use_count, r.name),
            reverse=True,
        )[:n]


_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_ROSTER_PATH = _DATA_DIR / "execution_agents" / "roster.json"

_agent_roster = AgentRoster(_ROSTER_PATH)


def get_agent_roster() -> AgentRoster:
    """Get the singleton roster instance."""
    return _agent_roster
