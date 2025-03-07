from abc import ABC, abstractmethod
from llm_providers import BaseLLM
from typing import Union, List
from prompts import MEMORY_PROMPT

class Agent(ABC):
    @abstractmethod
    def __init__(self, 
                 llm: Union[BaseLLM, List[BaseLLM]]):
        pass

    @abstractmethod
    def log(self) -> None: 
        pass

class MemoryAgent(Agent):
    """
    Write Conversational Data and Profile to PostGres Database. 
    """
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.prompt = MEMORY_PROMPT
    
    def __log__(self):
        pass