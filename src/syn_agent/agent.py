from langgraph.graph import StateGraph
from langgraph.types import interrupt
from abc import ABC, abstractmethod

from messages import Message
from tasks import Task
from llm_providers import BaseLLM
from typing import Union, Any, List

class Agent(ABC):

    @abstractmethod
    def __init__(self, 
                 llm: Union[BaseLLM, List[BaseLLM]]):
        pass

    @abstractmethod
    def __log__(self) -> None: 
        pass