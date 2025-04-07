from abc import ABC, abstractmethod
from llm_providers import BaseLLM
from typing import Union, List

class Agent(ABC):
    @abstractmethod
    def __init__(self, 
                 llm: Union[BaseLLM, List[BaseLLM]]):
        pass

    @abstractmethod
    def log(self) -> None: 
        pass