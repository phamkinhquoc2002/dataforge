from typing_extensions import TypedDict
from typing import Literal
from dataclasses import dataclass

@dataclass
class Message(TypedDict):
    """Message class."""
    role: Literal['user', 'system', 'assistant']
    content: str

@dataclass
class LogMessage(TypedDict):
    """LogMessage class."""
    type: Literal['ERROR', 'OUTPUT_MESSAGE', 'INFO']
    text: str