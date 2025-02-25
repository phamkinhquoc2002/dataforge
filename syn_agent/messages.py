from typing_extensions import TypedDict
from typing import Literal

class Message(TypedDict):
    role: Literal['user', 'system', 'assistant']
    content: str
