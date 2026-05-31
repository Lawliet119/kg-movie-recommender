"""
Conversation Session Manager — KGenSam Simplified (Level 2)

Tracks multi-turn conversation state for each user session:
- Accepted/rejected attributes (genre, director, actor preferences)
- Asked questions (avoid re-asking)
- Candidate movie set (narrows down each turn)
- Conversation policy state

Based on: KGenSam (Zhao et al., 2021) — dynamic user preference modeling
"""
import time
import uuid
import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConversationSession:
    """State for a single conversation session."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    # User preferences accumulated across turns
    accepted_attributes: dict[str, list[str]] = field(default_factory=lambda: {
        'genre': [],
        'person': [],     # directors + actors
        'year': [],
    })
    rejected_attributes: dict[str, list[str]] = field(default_factory=lambda: {
        'genre': [],
        'person': [],
        'year': [],
    })

    # Track what we've already asked
    asked_attributes: set[str] = field(default_factory=set)  # e.g. "genre:Sci-Fi", "person:Christopher Nolan"

    # Conversation counters
    turn_count: int = 0
    max_turns: int = 5

    # Candidate tracking
    candidate_movies: list[str] = field(default_factory=list)  # movie entity IDs
    recommended_movies: set[str] = field(default_factory=set)  # already recommended

    # Policy state
    should_recommend: bool = False
    last_question: Optional[dict] = None
    last_entropy: float = 1.0

    def touch(self):
        """Update last active timestamp."""
        self.last_active = time.time()

    def add_preference(self, attr_type: str, attr_value: str, accepted: bool):
        """Record a user preference response."""
        self.touch()
        key = f"{attr_type}:{attr_value}"
        self.asked_attributes.add(key)
        self.turn_count += 1

        if accepted:
            if attr_value not in self.accepted_attributes.get(attr_type, []):
                self.accepted_attributes.setdefault(attr_type, []).append(attr_value)
        else:
            if attr_value not in self.rejected_attributes.get(attr_type, []):
                self.rejected_attributes.setdefault(attr_type, []).append(attr_value)

    def get_all_accepted(self) -> list[tuple[str, str]]:
        """Get all accepted (type, value) pairs."""
        result = []
        for attr_type, values in self.accepted_attributes.items():
            for v in values:
                result.append((attr_type, v))
        return result

    def get_all_rejected(self) -> list[tuple[str, str]]:
        """Get all rejected (type, value) pairs."""
        result = []
        for attr_type, values in self.rejected_attributes.items():
            for v in values:
                result.append((attr_type, v))
        return result

    def is_asked(self, attr_type: str, attr_value: str) -> bool:
        """Check if we already asked about this attribute."""
        return f"{attr_type}:{attr_value}" in self.asked_attributes

    def to_dict(self) -> dict:
        """Serialize session state for API response."""
        return {
            'session_id': self.session_id,
            'turn_count': self.turn_count,
            'max_turns': self.max_turns,
            'accepted': self.accepted_attributes,
            'rejected': self.rejected_attributes,
            'asked_count': len(self.asked_attributes),
            'candidate_count': len(self.candidate_movies),
            'should_recommend': self.should_recommend,
            'last_entropy': round(self.last_entropy, 4),
        }


class SessionStore:
    """
    In-memory session store with auto-expiry.
    For production, replace with Redis or similar.
    """

    def __init__(self, expire_seconds: int = 1800):
        self._sessions: dict[str, ConversationSession] = {}
        self._expire_seconds = expire_seconds

    def create(self) -> ConversationSession:
        """Create a new conversation session."""
        session = ConversationSession()
        self._sessions[session.session_id] = session
        self._cleanup()
        logger.info(f"📝 Created session {session.session_id[:8]}...")
        return session

    def get(self, session_id: str) -> Optional[ConversationSession]:
        """Retrieve a session by ID."""
        session = self._sessions.get(session_id)
        if session and (time.time() - session.last_active) > self._expire_seconds:
            del self._sessions[session_id]
            logger.info(f"⏰ Session {session_id[:8]} expired")
            return None
        if session:
            session.touch()
        return session

    def delete(self, session_id: str):
        """Delete a session."""
        self._sessions.pop(session_id, None)

    def _cleanup(self):
        """Remove expired sessions."""
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if (now - s.last_active) > self._expire_seconds
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.info(f"🧹 Cleaned up {len(expired)} expired sessions")

    @property
    def active_count(self) -> int:
        return len(self._sessions)
