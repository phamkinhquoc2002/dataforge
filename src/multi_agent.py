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
from utils import prompt_initialize, one_shot_prompt, extract_valid_output, save_to_file
from agent import Agent
from datetime import datetime
import logging
import asyncio
from messages import Message

logging.basicConfig(level=logging.INFO)

class Feedback(BaseModel):
    """
    Feedback model for the multi-dialogue task.
    """
    approval: Literal['yes', 'no']
    feedback: Optional[str] = Field(None, max_length=300)

    @field_validator('feedback', mode='before')
    def check_feedback(cls, v: str):
        if len(v) > 300:
            raise ValueError('Your feedback should not exceed 300 characters')
        return v

class CurrentState(TypedDict):
    """
    Current state of the dialogue.
    """
    task: Task = Field(
        description="synthetic data generation mission"
    )
    human_feedback: Optional[Feedback] = Field(
        default=None, description="human-in-the-loop feedback"
        )
    response: Optional[str] = Field(
        default=None, description="response from the synthetic data generator"
        )
    conversations: List[dict] = Field(
        default_factory=list, description="short term memory"
        )
    retrieved_documents: Optional[List[Document]] = Field(
        default=None, description="retrieved documents from the retriever"
        )
    
    def get_len(self) -> int:
        """Return the number of conversation entries."""
        return len(self.conversations)

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
        self.output_path = output_path
        self.buffer_size = buffer_size
        self.retriever = retriever

        graph = StateGraph(CurrentState)
        graph.add_node('retrieve', self.retrieve)
        graph.add_node('fish_for_feedback', self.fish_for_feedback)
        graph.add_node('feedback_loop', self.human_in_the_loop)
        graph.add_node('data_generate', self.data_generate)
        graph.add_edge(START, 'retrieve')
        self.agent_flow = graph.compile(checkpointer=self.checkpointer)
    
    def fish_for_feedback(self, currentstate: CurrentState) -> Command:
        """
        Generate the first batch of output to fish for feedback from the user. 
        """ 
        conversations = currentstate.get('conversations')
        human_feedback = currentstate.get('human_feedback')
        task = currentstate.get('task')
        #Human Feedback Handling
        if human_feedback and human_feedback.get('approval') == 'yes':
            return Command(goto='data_generate')
        #Conversations and Prompt Handling
        turns = currentstate.get_len()
        self.log(mode='info', log_content=f"Total number of conversation turns up to this point: {turns}")    
        if currentstate['retrieved_documents']:
            task.grounded_knowledge = currentstate['retrieved_documents'][0]
        #Generate first batch of output
        if turns > 0:
            prompt = conversations
        else:
            prompt = prompt_initialize(task)
            self.log(mode='info', log_content=f"Prompt:{prompt}")
            conversations.extend(prompt)
            self.log(mode='info', log_content=f"Added prompt to conversation memory: {prompt}")
        try:
            response = self.llm(prompt)
            conversations.append({"role": "assistant", "content": response})
            self.log('info', f"Added response to conversation memory: {response}")
        except Exception as e:
            self.log('error', f"Error generating synthetic data: {e}")
            return Command(goto='__end__') 
        return Command(goto='feedback_loop', update={'response': response, 'conversations': conversations})
    
    def human_in_the_loop(self, currentstate: CurrentState) -> Command:
        """
        Human-in-the-loop feedback loop.
        """
        value = currentstate.get('response')
        conversations = currentstate.get('conversations')
        approval = interrupt(
            value="Do you approve?"
        )
        feedback = interrupt(
            value=f"This is my first batch of output, what do you think of it?\n{value}"
        )
        human_feedback = Feedback(approval=approval, feedback=feedback)
        conversations.append(
            {"role": "user", "content": feedback}
        )
        self.log(mode='info', log_content=f"Human in the loop Feedback: {feedback}")
        return Command(goto='fish_for_feedback', update={'human_feedback': human_feedback, 'conversations':conversations})
    
    def retrieve(self, currentstate: CurrentState) -> Command:
        """
        Retrieve necessary context to generate data pairs!
        """
        localized_context = currentstate['task'].localization
        retrieved_documents = self.retriever.invoke(localized_context)
        return Command(goto='fish_for_feedback', update={'retrieved_documents': retrieved_documents})
    
    async def data_generate(self, currentstate: CurrentState) -> Command:
        """
        Start generating the dataset.
        """
        response = currentstate.get('response')
        task = currentstate.get('task')
        retrieved_context = currentstate.get('retrieved_documents')

        num_tasks = min(task.batch_size, len(retrieved_context))
        optimized_prompts = await asyncio.gather(*[self.task_generate(
            task=task,
            i=i,
            response=response,
            retrieved_context=retrieved_context
        )  for i in range(num_tasks)])

        for prompt in optimized_prompts:
            self.log(mode='info', log_content=prompt)
            output = self.llm(prompt)
            self.response_memory.append(extract_valid_output(output))
            if len(self.response_memory) == self.buffer_size:
                self.save()
        return Command(goto='__end__')
    
    def task_generate(self, task: Task, i: int, response:str, retrieved_context: List[Document]) -> List[Message]:
        """Optimize every task and prompt!"""
        task.grounded_knowledge = retrieved_context[i]
        user_prompt = prompt_initialize(task)
        optimized_prompt = one_shot_prompt(user_prompt, response)
        return optimized_prompt
    
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
        save_to_file(self.response_memory, filename=self.output_path)

    def __call__(self) -> Any:
        pass