"""Save conversations and retrieve relevant messages for the current profile."""
import json
import sqlite3
from pathlib import Path
from uuid import uuid4
from langchain_core.messages import messages_from_dict, messages_to_dict
from app.memory.store import MemoryStore, MemoryStoreError, validate_identifier

CHAT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "chat_history.sqlite3"


class HistoryStore(MemoryStore):
    """Use the existing database connection and transaction handling."""

    def __init__(self, path: Path | None = None):
        super().__init__(path or CHAT_DB_PATH)
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS chat_attachments (
                    conversation_id TEXT PRIMARY KEY, snapshot_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    state_json TEXT
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL, role TEXT NOT NULL,
                    content TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
            """)

    def save_attachments(self, user_id, chat_id, snapshot):
        self.get_chat(user_id, chat_id)
        payload = json.dumps(snapshot, ensure_ascii=False)
        with self._connect() as connection:
            connection.execute("INSERT OR REPLACE INTO chat_attachments VALUES(?,?)", (chat_id, payload))

    def load_attachments(self, user_id, chat_id):
        self.get_chat(user_id, chat_id)
        with self._connect() as connection:
            row = connection.execute("SELECT snapshot_json FROM chat_attachments WHERE conversation_id=?", (chat_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def create(self, user_id: str) -> str:
        user_id = validate_identifier(user_id, "Profile ID")
        chat_id = uuid4().hex
        with self._connect() as connection:
            connection.execute("INSERT INTO conversations(id,user_id,title) VALUES(?,?,?)",
                               (chat_id, user_id, "New conversation"))
        return chat_id

    def get_chat(self, user_id: str, chat_id: str) -> dict:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute("SELECT * FROM conversations WHERE user_id=? AND id=?",
                                     (user_id, chat_id)).fetchone()
        if row is None:
            raise MemoryStoreError("Saved conversation was not found for this profile.")
        return dict(row)

    def list_chats(self, user_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id,title,updated_at FROM conversations WHERE user_id=? ORDER BY updated_at DESC,id DESC",
                (user_id,),
            ).fetchall()
        return [{"id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]

    def transcript(self, user_id: str, chat_id: str) -> list[dict]:
        self.get_chat(user_id, chat_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT role,content,metadata_json,created_at FROM chat_messages WHERE conversation_id=? ORDER BY id",
                (chat_id,),
            ).fetchall()
        return [{"role": r[0], "content": r[1], **json.loads(r[2]), "created_at": r[3]} for r in rows]

    def save_turn(self, user_id: str, chat_id: str, question: str, answer: str,
                  activity: list[str], sources: list[dict], state: dict) -> None:
        self.get_chat(user_id, chat_id)
        snapshot = json.dumps({"messages": messages_to_dict(state["messages"]), "model_calls": state["model_calls"]})
        metadata = json.dumps({"activity": activity, "sources": sources})
        with self._connect() as connection:
            connection.execute("INSERT INTO chat_messages(conversation_id,role,content) VALUES(?,?,?)",
                               (chat_id, "user", question))
            connection.execute("INSERT INTO chat_messages(conversation_id,role,content,metadata_json) VALUES(?,?,?,?)",
                               (chat_id, "assistant", answer, metadata))
            connection.execute("""
                UPDATE conversations SET state_json=?, updated_at=CURRENT_TIMESTAMP,
                title=CASE WHEN title='New conversation' THEN ? ELSE title END WHERE id=? AND user_id=?
            """, (snapshot, question[:60], chat_id, user_id))

    def completed_state(self, user_id: str, chat_id: str) -> dict | None:
        raw = self.get_chat(user_id, chat_id)["state_json"]
        if raw is None:
            return None
        data = json.loads(raw)
        return {"messages": messages_from_dict(data["messages"]), "model_calls": data["model_calls"]}

    def rename(self, user_id: str, chat_id: str, title: str) -> None:
        self.get_chat(user_id, chat_id)
        title = title.strip()
        if not title or len(title) > 100:
            raise MemoryStoreError("Chat title must contain 1–100 characters.")
        with self._connect() as connection:
            connection.execute("UPDATE conversations SET title=?,updated_at=CURRENT_TIMESTAMP WHERE id=? AND user_id=?",
                               (title, chat_id, user_id))

    def delete(self, user_id: str, chat_id: str) -> None:
        self.get_chat(user_id, chat_id)
        with self._connect() as connection:
            connection.execute("DELETE FROM chat_attachments WHERE conversation_id=?", (chat_id,))
            connection.execute("DELETE FROM chat_messages WHERE conversation_id=?", (chat_id,))
            connection.execute("DELETE FROM conversations WHERE id=? AND user_id=?", (chat_id, user_id))

    def export_markdown(self, user_id: str, chat_id: str) -> str:
        chat = self.get_chat(user_id, chat_id)
        lines = ["# " + chat["title"], "", "Profile: " + user_id, ""]
        for message in self.transcript(user_id, chat_id):
            lines.extend(["## " + ("You" if message["role"] == "user" else "Agent"),
                          "", message["created_at"] + " UTC", "", message["content"], ""])
            for event in message.get("activity", []):
                lines.extend(["- " + event, ""])
            for source in message.get("sources", []):
                label = (source["document"] + ", " + ("p. " if source.get("type") == "pdf" else source.get("location", "section") + " ") + str(source["page"])) if source.get("type") in {"pdf", "document"} else source.get("url", source.get("document", "attachment"))
                lines.extend(["Source: " + label, ""])
        return "\n".join(lines)

    def previous_user_name(self, user_id: str) -> str | None:
        """Find explicit name statements in this profile's user messages."""
        import re
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT m.content FROM chat_messages m JOIN conversations c "
                "ON c.id=m.conversation_id WHERE c.user_id=? AND m.role='user' "
                "ORDER BY m.id DESC LIMIT 200", (user_id,),
            ).fetchall()
        for (content,) in rows:
            match = re.fullmatch(
                r"(?:please\s+)?(?:remember(?:\s+that)?\s+)?my name is\s+"
                r"([A-Za-z][A-Za-z '\-]{0,79})[.!]?", content.strip(), re.I,
            )
            if match:
                return match.group(1).strip()
        return None


    def relevant_history(self, user_id: str, question: str, exclude_chat: str) -> list[dict]:
        """Search saved messages locally and return a small set of relevant excerpts."""
        import re
        from collections import Counter
        ignored = {"what", "which", "that", "this", "have", "with", "your", "about",
                   "remember", "please", "tell", "does", "from", "were"}
        tokens = set(re.findall(r"[a-z0-9]{3,}", question.lower())) - ignored
        candidates = []
        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT m.conversation_id,m.role,m.content,m.created_at,m.id "
                "FROM chat_messages m JOIN conversations c ON c.id=m.conversation_id "
                "WHERE c.user_id=? AND c.id<>? ORDER BY m.id DESC",
                (user_id, exclude_chat),
            )
            # Read saved messages one row at a time to limit memory use.
            for chat_id, role, content, created, message_id in cursor:
                words = Counter(re.findall(r"[a-z0-9]{3,}", content.lower()))
                score = sum(min(words[word], 3) for word in tokens)
                if len(candidates) < 6 or score:
                    candidates.append((score, message_id, {
                        "role": role, "text": content[:1000], "created_at": created,
                    }))
                if len(candidates) > 40:
                    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
                    candidates = candidates[:20]
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        chosen = sorted(candidates[:8], key=lambda item: item[1])
        return [item[2] for item in chosen]


    def history_overview(self, user_id: str) -> dict:
        """Return a short overview of questions from this profile's conversations."""
        with self._connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) FROM chat_messages m JOIN conversations c ON c.id=m.conversation_id "
                "WHERE c.user_id=? AND m.role='user'", (user_id,),
            ).fetchone()[0]
            rows = connection.execute(
                "SELECT c.title,m.content,m.created_at FROM chat_messages m "
                "JOIN conversations c ON c.id=m.conversation_id "
                "WHERE c.user_id=? AND m.role='user' ORDER BY m.id DESC LIMIT 100",
                (user_id,),
            ).fetchall()
        return {"total_saved_questions": total, "included_questions": len(rows),
                "coverage": "all saved questions" if total <= 100 else "latest 100 saved questions",
                "questions": [{"chat": title[:80], "question": content[:240], "created_at": created}
                              for title, content, created in reversed(rows)]}
