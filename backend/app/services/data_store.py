import json
import threading
import uuid
from pathlib import Path
from typing import Any

from app.core.config import get_settings


class SessionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._base_dir = get_settings().data_dir / "sessions"
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def create_session(
        self,
        anchors: list[dict[str, Any]],
        products: list[dict[str, Any]],
        uploaded_files: list[str],
    ) -> str:
        session_id = uuid.uuid4().hex
        payload = {
            "session_id": session_id,
            "anchors": anchors,
            "products": products,
            "uploaded_files": uploaded_files,
        }
        with self._lock:
            self._sessions[session_id] = payload
        self._persist(session_id, payload)
        return session_id

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            cached = self._sessions.get(session_id)
        if cached:
            return cached
        return self._load(session_id)

    def _session_path(self, session_id: str) -> Path:
        return self._base_dir / f"{session_id}.json"

    def _persist(self, session_id: str, payload: dict[str, Any]) -> None:
        path = self._session_path(session_id)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _load(self, session_id: str) -> dict[str, Any] | None:
        path = self._session_path(session_id)
        if not path.exists():
            return None
        loaded = json.loads(path.read_text(encoding="utf-8"))
        with self._lock:
            self._sessions[session_id] = loaded
        return loaded


session_store = SessionStore()
