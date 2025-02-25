from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph, START
from langgraph.types import interrupt, Command
from pydantic import BaseModel, Field,field_validator
from typing import Literal, List, Union, Any, Optional
from typing_extensions import TypedDict
from llm_providers import BaseLLM
from tasks import Task
from utils import prompt_initialize
from agent import Agent
import logging
from datetime import datetime
from messages import Message

logging.basicConfig(level=logging.INFO)

class Feedback(BaseModel):
    """
    Feedback model for the multi-dialogue task.
    """
    approval: Literal['yes', 'no']
    feedback: Optional[str] = Field(None, max_length=300)

    @field_validator('feedback')
    def check_feedback(cls, v: str):
        if len(v) > 300:
            raise ValueError('Your feedback should not exceed 300 characters')
        return v

class CurrentState(TypedDict):
    """
    Current state of the dialogue.
    """
    task: Task
    human_feedback: Optional[Feedback] = Field(None, description="human-in-the-loop feedback")
    response: Optional[str] = Field(None, description="response from the synthetic data generator")

class SyntheticDataGenerator(Agent):
    """
    Synthetic data generator agent.
    """
    def __init__(self, 
                 llm: Union[BaseLLM, List[BaseLLM]],
                 buffer_size: int = 10):
        self.llm = llm
        self.checkpointer = MemorySaver()
        self.long_term_memory = InMemoryStore()
        self.conversations: List[dict] = []
        self.logger = logging.getLogger(__name__)
        self.conversations: List[dict] = []
        graph = StateGraph(CurrentState)
        graph.add_node('generate', self.generate)
        graph.add_node('feedback_loop', self.human_in_the_loop)
        graph.add_edge(START, 'generate')
        self.agent_flow = graph.compile(checkpointer=self.checkpointer)
    
    def get_len(self) -> int:
        """Get the number of conversations up to this point!"""
        return len(self.conversations)
    
    def generate(self, currentstate: CurrentState) -> Command:
        """
        Generate synthetic data based on the current state of the dialogue.
        """
        task = currentstate['task']
        turns = self.get_len()
        if turns >0:
            prompt = self.conversations
        else:
            prompt = prompt_initialize(task)
            self.conversations.extend(prompt)
            self.log(mode='info', log_content=f"""Added to conversation memory: {prompt}""")
        try:
            response = self.llm(prompt)
            self.conversations.append(
                {"role": "assistant", "content": response}
            )
            self.log('info', "Saved the first conversation to short-term-memory")
        except Exception as e:
            error = f"Error generating synthetic data: {e}"
            self.logger.error(error)
            return Command(goto='__end__', update={'error': error})
        if currentstate['human_feedback']:
            approval = currentstate['human_feedback'].approval
            if approval == 'yes':
                return Command(goto='__end__', update={'response': response})
            elif approval == 'no':
                return Command(goto='feedback_loop', update={'response': response})
        elif currentstate['human_feedback'] is None:
            return Command(goto='feedback_loop', update={'response': response})
    
    def human_in_the_loop(self, currentstate: CurrentState) -> Command:
        """
        Human-in-the-loop feedback loop.
        """
        value = currentstate['response']
        approval = interrupt(
            value=f"Do you approve?"
        )
        feedback = interrupt(
            value=f"What do you think about this output?\n{value}"
        )
        human_feedback = Feedback(approval=approval, feedback=feedback)
        self.conversations.append(
            {"role": "user", "content": feedback}
        )
        feedback_log = f"Human in the loop Feedback: {feedback}"
        self.log(mode='info', log_content=feedback_log)
        return Command(goto='generate', update={'human_feedback': human_feedback})

    def log(self, mode: str, log_content: str) -> None:
        """Log messages with context."""
        if mode == 'info':
            self.logger.info(log_content)
        elif mode == 'error':
            self.logger.error(log_content)

    def save_log(self) -> None:
        """Saved the conversations to a log file."""
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(f'running_{date}.log', 'w') as f:
            f.write('\n'.join(self.conversations))
    
    def __call__(self) -> Any:
        pass

    def save_files(self) -> Any:
        pass