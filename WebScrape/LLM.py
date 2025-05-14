# File: chat_session.py
from abc import ABC, abstractmethod

class ChatSession(ABC):
    """
    Abstraction for a chat-based LLM session.  
    send() takes a prompt string and returns a tuple (response_text, error_code).
    error_code is None on success, or an exception name/description on failure.
    """

    @abstractmethod
    def send(self, prompt: str) -> tuple[str, str | None]:
        pass