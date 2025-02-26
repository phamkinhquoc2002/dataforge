from langgraph.checkpoint.memory import MemorySaver
from langchain.schema import Document
from langchain.retrievers import BM25Retriever
from langgraph.graph import StateGraph, START
from langgraph.types import interrupt, Command
from pydantic import BaseModel, Field,field_validator
from typing import Literal, List, Union, Any, Optional
from typing_extensions import TypedDict
from llm_providers import BaseLLM
from tasks import Task
from utils import prompt_initialize, extract_valid_output, save_to_file
from agent import Agent
from datetime import datetime
import logging

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
    retrieved_documents: Optional[List[Document]] = Field(None, description="retrieved documents from the retriever")

class SyntheticDataGenerator(Agent):
    """
    Synthetic Data Generation Agentic System.

    """
    def __init__(self, 
                 llm: Union[BaseLLM, List[BaseLLM]],
                 retriever: BM25Retriever,
                 output_path: str,
                 buffer_size: int = 10):
        self.llm = llm
        self.checkpointer = MemorySaver()
        self.response_memory = []
        self.conversations: List[dict] = []
        self.logger = logging.getLogger(__name__)
        self.conversations: List[dict] = []
        self.output_path = output_path
        self.buffer_size = buffer_size
        self.retriever = retriever
        self.retrieved_index = 0

        graph = StateGraph(CurrentState)
        graph.add_node('retrieve', self.retrieve)
        graph.add_node('generate', self.generate)
        graph.add_node('feedback_loop', self.human_in_the_loop)
        graph.add_edge(START, 'retrieve')
        self.agent_flow = graph.compile(checkpointer=self.checkpointer)
    
    def get_len(self) -> int:
        """Get the number of conversations up to this point!"""
        return len(self.conversations)
    
    def generate(self, currentstate: CurrentState) -> Command:
        """
        Generate synthetic data based on the current state of the dialogue.
        """
        turns = self.get_len()
        self.log(mode='info', log_content=f"Total number of conversation turns up to this point: {turns}")
        task = currentstate['task']

        if not currentstate.get('retrieved_documents') or len(currentstate['retrieved_documents']) <= self.retrieved_index:
            return Command(goto='retrieve')
        
        task.grounded_knowledge = currentstate['retrieved_documents'][self.retrieved_index]
        if turns > 0:
            prompt = self.conversations
        else:
            prompt = prompt_initialize(task)
            self.log(mode='info', log_content=f"Prompt:{prompt}")
            self.conversations.extend(prompt)
            self.log(mode='info', log_content=f"Added to conversation memory: {prompt}")
        try:
            response = self.llm(prompt)
            self.conversations.append({"role": "assistant", "content": response})
            self.log('info', f"Added to conversation memory: {response}")
        except Exception as e:
            error = f"Error generating synthetic data: {e}"
            self.log('error', f"Error generating synthetic data: {e}")
            return Command(goto='__end__', update={'error': error})
        
        if currentstate.get('human_feedback'):
            approval = currentstate['human_feedback'].approval
            if approval == 'yes':
                parsed_responses = extract_valid_output(response)
                self.response_memory.extend(parsed_responses)
                self.retrieved_index +=1
                return Command(goto='retrieve')
            elif approval == 'no':
                return Command(goto='feedback_loop', update={'response': response})
            
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
        self.log(mode='info', log_content=f"Human in the loop Feedback: {feedback}")
        return Command(goto='generate', update={'human_feedback': human_feedback})
    
    def retrieve(self, currentstate: CurrentState) -> Command:
        """
        Retrieve necessary context to generate data pairs!
        """
        localized_context = currentstate['task'].localization
        retrieved_documents = self.retriever.invoke(localized_context)
        return Command(goto='generate', update={'retrieved_documents': retrieved_documents})
    
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

    def save(self) -> None:
        """Saved to directory."""
        if len(self.response_memory) == self.buffer_size:
            save_to_file(self.response_memory, filename=self.output_path)

    def __call__(self) -> Any:
        pass