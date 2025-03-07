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
from utils import prompt_initialize, one_shot_prompt, extract_valid_output, save_to_file, log_message, document_format
from agent import Agent
from datetime import datetime
import asyncio
from messages import Message

class Feedback(BaseModel):
    """
    Feedback model for the multi-dialogue task.
    """
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
    approval: Optional[Literal["yes", "no"]] = Field(
        default=None, description="approval of the first response"
    )
    human_feedback: Optional[Feedback] = Field(
        default=None, description="human-in-the-loop feedback"
        )
    response: Optional[str] = Field(
        default=None, description="response from the synthetic data generator"
        )
    conversations: List[Message] = Field(
        default=[], description="short term memory"
        )
    retrieved_documents: Optional[List[str]] = Field(
        default=None, description="retrieved documents from the retriever"
        )

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
        self.output_path = output_path
        self.buffer_size = buffer_size
        self.retriever = retriever

        graph = StateGraph(CurrentState)
        graph.add_node('retrieve', self.retrieve)
        graph.add_node('fish_for_feedback', self.fish_for_feedback)
        graph.add_node('human_approval', self.approve)
        graph.add_node('feedback_loop', self.human_in_the_loop)
        graph.add_node('data_generate', self.data_generate)
        graph.add_edge(START, 'retrieve')
        self.agent_flow = graph.compile(checkpointer=self.checkpointer)
    
    def fish_for_feedback(self, currentstate: CurrentState) -> Command:
        """
        Generate the first batch of output to fish for feedback from the user. 
        """ 
        conversations = currentstate.get('conversations')
        task = currentstate.get('task')
        turns = len(conversations)
        log_message(
            {
                "type":"INFO",
                "text": f"Total number of conversation turns up to this point: {turns}"
            }
        )   

        if currentstate['retrieved_documents']:
            task.grounded_knowledge = currentstate['retrieved_documents'][0]

        if turns > 0:
            log_message(
                {
                    "type": "INFO",
                    "text": f"FEEDBACK CHECK: {conversations[-1]['content']}"
                }
            )
            prompt = conversations
        else:
            prompt = prompt_initialize(mode="fish", task=task)
            conversations.extend(prompt)
            log_message(
                {
                    "type": "OUTPUT_MESSAGE",
                    "text": f"\n---------FISH_FOR_FEEDBACK---------\n\nSystem Prompt:{conversations[0]['content']}\nUser Prompt:{conversations[1]['content']}."
                }
            )

        try:
            response = self.llm(prompt)
            conversations.append({"role": "assistant", "content": response})
            log_message(
                {
                    "type": "OUTPUT_MESSAGE",
                    "text": f"\n---------FISH_FOR_FEEDBACK---------\n\nResponse:{response}"
                }
            )
        except Exception as e:
            log_message(
                {
                    "type": "ERROR",
                    "text": f"ERROR generating synthetic data: {e}"
                }
            )
            raise e
        return Command(goto='human_approval', update={'response': response, 'conversations': conversations})
    
    def approve(self, currentstate: CurrentState) -> Command:
        """
        Approve the first response.
        """
        approval = interrupt(
            value="Do you approve?"
        )
        if approval == "yes":
            return Command(goto='data_generate',  update={'approval': approval})
        elif approval == "no":
            log_message(
                {
                    "type":"INFO",
                    "text": "User disapproved the response. Proceeding with refinement."
                }
            )
            return Command(goto='feedback_loop',  update={'approval': approval})

    def human_in_the_loop(self, currentstate: CurrentState) -> Command:
        """
        Human-in-the-loop feedback loop.
        """
        conversations = currentstate.get('conversations')
        feedback = interrupt(
            value=f"Any Feedback to improve the answer?\n"
        )
        human_feedback = Feedback(feedback=feedback)
        conversations.append(
            {"role": "user", "content": feedback}
        )
        log_message(
            {
                "type":"OUTPUT_MESSAGE",
                "text": f"\n---------HUMAN_IN_THE_LOOP---------\n\nUser Feedback{feedback}"
            }
        )
        return Command(goto='fish_for_feedback', update={'human_feedback': human_feedback.model_dump(), 'conversations':conversations})
    
    def retrieve(self, currentstate: CurrentState) -> Command:
        """
        Retrieve necessary context to generate data pairs.
        """
        localized_context = currentstate['task'].localization
        raw_documents = self.retriever.invoke(localized_context)
        if not raw_documents:
            log_message(
                {
                    "type": "ERRPR",
                    "text": "No documents retrieved. Proceeding without context."
                }
            )
            return Command(goto='__end__')
        formated_documents = document_format(raw_documents)
        return Command(goto='fish_for_feedback', update={'retrieved_documents': formated_documents})
    
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

        for num, prompt in enumerate(optimized_prompts):
            log_message(
                {
                    "type":"INFO",
                    "text": f"{prompt[1]['content'][:200]}"
                }
            )
            output = await self.llm(prompt)
            log_message(
                {
                    "type":"OUTPUT_MESSAGE",
                    "text": f"\n---------DATA_GENERATE---------\n\nSample output of batch {num}:{prompt[:200]}"
                }
            )
            self.response_memory.append(extract_valid_output(output))
            if len(self.response_memory) == self.buffer_size:
                self.save()
        return Command(goto='__end__')
    
    async def task_generate(self, 
                            task: Task, 
                            i: int, 
                            response:str, retrieved_context: List[Document]) -> List[Message]:
        """Optimize every task and prompt!"""
        task.grounded_knowledge = retrieved_context[i]
        user_prompt = prompt_initialize(mode='real', task=task)
        optimized_prompt = one_shot_prompt(user_prompt, response)
        return optimized_prompt

    def log(self) -> None:
        """Saved the conversations to a log file."""
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(f'running_{date}.log', 'w') as f:
            f.write('\n'.join(self.conversations))

    def save(self) -> None:
        """Saved to directory."""
        save_to_file(self.response_memory, filename=self.output_path)

    def __call__(self) -> Any:
        pass