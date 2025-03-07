import re
import json
import os
from prompts import SFT, DPO, CONVERSATION, SYSTEM_PROMPT
from messages import Message, LogMessage
from tasks import Task
from typing import List, Literal
from pypdf import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from copy import deepcopy

def log_message(log: LogMessage) -> None:
    """
    Log message beautifully.

    Parameters:
        log_type (LogMessage): Type of Logging.
        text: Logging Message
    """
    console = Console()
    formatted_text = Text()

    formatted_text.append(log['text'], style="white")
    if log["type"]== 'DEBUG':
        border_style = 'red'
    elif log["type"] == 'OUTPUT_MESSAGE':
        border_style = 'green'
    elif log["type"] == 'INFO':
        border_style = 'yellow'
    console.print(Panel(formatted_text, title=log["type"], 
                        title_align="center", 
                        border_style=border_style))

def key_map(dictionary: dict, 
            key:str, 
            default=None) -> str:
    """
    Retrieve the value from 'dictionary' for the given 'key'.
    
    Parameters:
        dictionary (dict): The dictionary to search.
        key: The key to look for.
        default: Value to return if the key isn't found (defaults to None).
        
    Returns:
        The value associated with 'key' if it exists, otherwise 'default'.
    """
    return dictionary.get(key, default)

def user_prompt_initialize(
        mode:Literal['fish', 'real'], 
        task: Task) -> str:
    """
    Generate a user prompt for the synthetic data generator.
    
    Parameters:
        task (Task): The task for which to generate a prompt.
        
    Returns:
        The user prompt.
    """
    tasks = {'sft': SFT, 'dpo': DPO, 'multi-dialogue': CONVERSATION}
    task_format = key_map(tasks, task.task_name)
    grounded = task.grounded_knowledge
    if mode == 'fish':
        num_of_data=2
    elif mode == 'real':
        num_of_data = task.rows_per_batch
    if task.task_description:
        task_description = f"Additional Dataset Info: {task.task_description}"
    else:
        task_description = ""
    if task.language:
        language = f"entirely in {task.language}"
    else:
        language = ""
    return f"You are tasked to help me generate a dataset of {num_of_data} rows {language}, based entirely on the following context:{grounded}\n{task_format}\n{task_description}\n"

def prompt_validator(func):
    """
    Validate the prompt format for data generator pipeline.
    """
    def wrapper(*args, **kwargs):
        log_message(
            {
                "type":"INFO",
                "text":f"Calling {func.__name__}"
            }
        )
        docs = func(*args, **kwargs)

        if len(docs) < 2:
            log_message(
                {
                    "type" : "ERROR", 
                    "text": "Invalid return format: must be a list with at least two messages."
                }
            )
            raise ValueError("The returned list must contain at least two messages.")
        
        if docs[0]["role"] != "system":
            log_message(
                {
                    "type":"ERROR",
                    "text":"First message role is incorrect."
                }
            )
            raise ValueError("The first message must have the role 'system'.")
        
        if docs[1]["role"] != "user":
            log_message(
                {
                    "type":"ERROR",
                    "text": "Second message role is incorrect."
                }
            )
            raise ValueError("The second message must have the role 'user'.")
        
        log_message(
            {
                "type": "INFO",
                "text": f"""System Prompt: {docs[0]['content'][:100]}\nUser Prompt: {docs[1]['content'][:100]}\n"""
            }
        )
        return docs
    return wrapper

@prompt_validator
def prompt_initialize(
        mode:Literal['fish', 'real'], 
        task: Task
) -> List[Message]:
    """
    Initialize a conversation with the synthetic data generator.

    Parameters:
        mode (TaskLiteral['fish', 'real']): Prompt generation mode.
        task (Task): The task for which to generate a prompt.
    
    Returns:
        A list of message containing the system prompt and user prompt.
    """
    messages = []
    system_prompt= {"role": "system", 
                    "content": SYSTEM_PROMPT}
    user_prompt = {"role": "user", 
                   "content": user_prompt_initialize(mode=mode, task=task)}
    messages.append(system_prompt)
    messages.append(user_prompt)
    return messages

@prompt_validator
def one_shot_prompt(user_prompt:List[Message], 
                    response: str) -> List[Message]:
    """
    Create a behavioral one-shot prompt to adapt the llm to user's preferred answer.

    Parameters:
        user_prompt (List[Message]): User original prompt.
        response (str): Approved one-shot example.

    Returns:
        Prompt after being optimized.
    """
    user_prompt_copy = deepcopy(user_prompt)
    user_prompt_copy[1]['content'] += f"\nExample:\n{response}"
    log_message(
        {
            "type":"OUTPUT_MESSAGE",
            "text":user_prompt_copy[1]['content']
        }
    )
    return user_prompt_copy
    
def extract_valid_output(output: str) -> List[dict]:
    """
    Extract the valid output from the response.
    
    Parameters:
        output (str): The response from the synthetic data generator.
        
    Returns:
        The valid output.
    """
    try:
        pattern = re.compile(r"\[.*?\]", re.DOTALL)
        match = pattern.search(output)
        match_str = match.group()
        parsed = json.loads(match_str)
        if isinstance(parsed, list):
            return parsed
    except Exception as e:
        log_message(
            {
                "type":"ERROR",
                "text":"The format is invalid to extract. You must return the exact format as specified in the prompt"
            }
        )
        raise ValueError("Failed to extract valid JSON output.")   

def save_to_file(output: List[dict], 
                 filename: str) -> None:
    """
    Save the output to a file.
    
    Parameters:
        output (List[dict]): The output to save.
        filename (str): The name of the file to save to.
    """
    try:
        with open(filename, 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError as e:
        log_message(
            {
                "type":"INFO",
                "text":"Can't locate the file in your directory. Return an empty list instead!"
            }
        )
        existing_data = []

    existing_data.extend(output)
    with open(filename, 'w') as f:
        json.dump(existing_data, f)

def chunk_parser(func):
    """
    Decorator to split PDF content into smaller, manageable chunks.
    """
    def wrapper(*args, **kwargs):
        log_message(
            {
            "type" : "INFO", 
            "text": f"Calling {func.__name__}"
            }
        )

        docs = func(*args, **kwargs)
        documents = [Document(page_content=doc) for doc in docs]
        log_message(
            {
                "type":"INFO", 
                "text":f"Loading {len(documents)} pieces of context!"
            }
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000,
            chunk_overlap=200
        )
        splitted_texts = text_splitter.split_documents(documents)
        
        if splitted_texts:
            preview = splitted_texts[-1].page_content[:100]
            log_message(
                {
                    "type": "INFO", 
                    "text":f"Split successful. Last chunk preview:\n{preview}..."
                }
            )
        return splitted_texts
    return wrapper

@chunk_parser
def pdf_parser(path: str) -> List[Document]:
    """
    Parse PDF file and extract text content.
    
    Parameters:
        path (str): Path to the PDF file.
        
    Returns:
        list: List of extracted text from PDF pages.
    """
    documents = []
    if os.path.exists(path):
        try:
            pdf = PdfReader(path)
            for page in pdf.pages:
                page_text = page.extract_text()
                documents.append(page_text)
            if documents:
                log_message(
                    {
                        "type":"INFO",
                        "text": f"Extracted {len(documents)} pages. First page preview:\n{documents[0][:100]}..."
                    }
                )
            else:
                log_message(
                    {
                        "type":"INFO",
                        "text":"PDF file contains no extractable text"
                    }
                )
        except Exception as e:
            log_message(
                {
                    "type": "ERROR", 
                    "text":f"Error parsing PDF: {str(e)}"
                }
            )
            raise ValueError("Error parsing PDF file")
    else:
        log_message(
            {
                "type":"INFO",
                "text": f"PDF file not found at path: {path}"
            }
        )
    return documents

def document_format(retrieved_documents: List[Document]) -> List[str]:
    """
    Re-format the number of documents retrieved.

    Parameters:
        retrieved_documents (List[Document]): Retrieved documents from the retriever.
        
    Returns:
        List of formatted documents.
    """
    formatted_docs = []
    for doc in retrieved_documents:
        text = ''.join([f'\n- {doc.page_content}'])
        formatted_docs.append(text)
    log_message(
        {
            "type":"OUTPUT_MESSAGE", 
            "text": f"FORMATTED DOCUMENTS PREVIEW: {formatted_docs[0][:200]}"
        }
    )
    return formatted_docs