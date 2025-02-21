from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command
from pydantic import BaseModel, Field,field_validator
from typing import Literal, List, Union, Any
from llm_providers import BaseLLM
from tasks import Task
from utils import prompt_initialize
from agent import Agent
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

class Feedback(BaseModel):
    """
    Feedback model for the multi-dialogue task.
    """
    next_node: Literal['yes', 'no']
    feedback: str = Field(None, max_length=300)
    llm_response: str

    @field_validator('feedback')
    def check_feedback(cls, v: str):
        if len(v) > 300:
            raise ValueError('Your feedback should not exceed 300 characters')
        return v

class CurrentState(StateGraph):
    """
    Current state of the dialogue.
    """
    task: Task
    human_feedback: Feedback
    response: str

class SyntheticDataGenerator(Agent):
    """
    Synthetic data generator agent.
    """
    def __init__(self, 
                 llm: Union[BaseLLM, List[BaseLLM]],
                 buffer_size: int = 10):
        self.llm = llm
        self.short_term_memory = MemorySaver()
        self.long_term_memory = InMemoryStore()
        self.logger = logging.getLogger(__name__)
        self.log_memory = []
        self.buffer_size = buffer_size
        
    def generate(self, currentstate: CurrentState) -> Command:
        """
        Generate synthetic data based on the current state of the dialogue.
        """
        task = currentstate['task']
        prompt = prompt_initialize(task)
        try:
            response = self.llm.generate(prompt)
        except Exception as e:
            error = f"Error generating synthetic data: {e}"
            self.logger.error(error)
            self.log_memory.append(error)
        return Command(goto='human_feedback', update={'response': response})
    
    def human_in_the_loop(self, currentstate: CurrentState) -> Command:
        value = currentstate['response']
        feedback = interrupt(
            value=f"What do you think about this output?\n{value}"
        )
        feedback_log = f"Human in the loop Feedback: {feedback}"
        self.log(mode='info', feedback_log=feedback_log)
        return Command(goto='generate', update={'human_feedback': feedback})

    def log(self, mode: str, log_content: str) -> None:
        if mode == 'info':
            self.logger.info(log_content)
        elif mode == 'error':
            self.logger.error(log_content)
        self.log_memory.append(log_content)

    def save_log(self) -> None:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(f'running_{date}.log', 'w') as f:
            f.write('\n'.join(self.log_memory))

    def __call__(self) -> Any:
        pass