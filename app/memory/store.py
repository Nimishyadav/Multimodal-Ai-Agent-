"""Store user preferences in a local SQLite database."""
import logging
from contextlib import contextmanager
from collections.abc import Iterator
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "memory.sqlite3"
MAX_PREFERENCES = 20
MAX_VALUE_LENGTH = 500


class MemoryStoreError(ValueError):
    """Safe storage error suitable for a user-facing message."""


@dataclass(frozen=True)
class Preference:
    key: str
    value: str
    updated_at: str


def validate_identifier(value: str, label: str) -> str:
    value = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise MemoryStoreError(f"{label} must contain 1–64 letters, numbers, underscores or hyphens.")
    return value


class MemoryStore:
    def __init__(self, path: Path = DEFAULT_DB_PATH) -> None:
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS preferences (
                        user_id TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (user_id, key)
                    )
                """)
        except (sqlite3.Error, OSError):
            raise MemoryStoreError("Memory storage could not be opened.") from None

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def save(self, user_id: str, key: str, value: str) -> None:
        user_id = validate_identifier(user_id, "Profile ID")
        key = validate_identifier(key, "Preference key")
        value = value.strip()
        if not value or len(value) > MAX_VALUE_LENGTH:
            raise MemoryStoreError(f"Preference value must contain 1–{MAX_VALUE_LENGTH} characters.")
        try:
            with self._connect() as connection:
                # Check the count and save in one transaction to avoid conflicting writes.
                connection.execute("BEGIN IMMEDIATE")
                count = connection.execute(
                    "SELECT COUNT(*) FROM preferences WHERE user_id = ?", (user_id,)
                ).fetchone()[0]
                exists = connection.execute(
                    "SELECT 1 FROM preferences WHERE user_id = ? AND key = ?", (user_id, key)
                ).fetchone()
                if count >= MAX_PREFERENCES and not exists:
                    raise MemoryStoreError(f"A profile can have at most {MAX_PREFERENCES} saved preferences.")
                connection.execute("""
                    INSERT INTO preferences(user_id, key, value, updated_at)
                    VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                    ON CONFLICT(user_id, key) DO UPDATE SET
                    value = excluded.value, updated_at = excluded.updated_at
                """, (user_id, key, value))
        except sqlite3.Error:
            logger.warning("Preference save failed.")
            raise MemoryStoreError("Could not save the preference. Please try again.") from None

    def list_preferences(self, user_id: str) -> list[Preference]:
        user_id = validate_identifier(user_id, "Profile ID")
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT key, value, updated_at FROM preferences WHERE user_id = ? ORDER BY key",
                    (user_id,),
                ).fetchall()
            return [Preference(*row) for row in rows]
        except sqlite3.Error:
            logger.warning("Preference retrieval failed.")
            raise MemoryStoreError("Could not read saved preferences. Please try again.") from None

    def get(self, user_id: str, key: str) -> str | None:
        key = validate_identifier(key, "Preference key")
        return next((p.value for p in self.list_preferences(user_id) if p.key == key), None)

    def delete(self, user_id: str, key: str) -> bool:
        user_id = validate_identifier(user_id, "Profile ID")
        key = validate_identifier(key, "Preference key")
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    "DELETE FROM preferences WHERE user_id = ? AND key = ?", (user_id, key)
                )
            return cursor.rowcount > 0
        except sqlite3.Error:
            logger.warning("Preference deletion failed.")
            raise MemoryStoreError("Could not delete the preference. Please try again.") from None