import re
import json
import os
import functools
import time
from src.prompts import SFT, DPO, CONVERSATION, SYSTEM_PROMPT
from src.messages import Message, LogMessage
from src.tasks import Task
from typing import List, Literal
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker
from langchain.schema import Document
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
    if log["type"]== 'ERROR':
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
    
def extract_valid_output(output: str):
    """
    Extract the valid JSON output from a possibly messy response.

    Parameters:
        output (str): The response from the synthetic data generator.

    Returns:
        The valid parsed object (list or dict), or None if failed.
    """
    try:
        parsed = json.loads(output)
        if isinstance(parsed, (list, dict)):
            return parsed
    except json.JSONDecodeError:
        pass  
    try:
        start = output.find('[')
        end = output.rfind(']') + 1
        if start != -1 and end != -1:
            json_fragment = output[start:end]
            parsed = json.loads(json_fragment)
            if isinstance(parsed, (list, dict)):
                return parsed
    except Exception as e:
        log_message({"type": "ERROR", "text": f"Manual extraction failed: {e}"})

    try:
        matches = re.findall(r'\[\s*(?:.|\n)*?\]', output)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, (list, dict)):
                    return parsed
            except json.JSONDecodeError:
                continue
    except Exception as e:
        log_message({"type": "ERROR", "text": f"Regex fallback failed: {e}"})

    log_message({"type": "ERROR", "text": "Could not extract valid JSON output"})
    return None

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

def pdf_parser(path: str) -> List[Document]:
    """
    Parse PDF file and extract text content.
    
    Parameters:
        path (str): Path to the PDF file.
    Returns:
        List[Document]: List of Document objects containing extracted text from the PDF pages.
    """
    chunker = HybridChunker(max_tokens=4000)
    loader = DoclingLoader(path, chunker=chunker)

    if os.path.exists(path):
        try:
            pdf = loader.load()
            log_message(
                    {
                        "type":"INFO",
                        "text": f"Extracted {len(pdf)} pages. First page preview:\n\n{pdf[0].page_content}..."
                    }
                )
        except Exception as e:
            log_message(
                {
                    "type": "ERROR", 
                    "text":f"Error parsing PDF:\n\n{str(e)}"
                }
            )
            raise ValueError("Error parsing PDF file")
    else:
        log_message(
            {
                "type":"INFO",
                "text": f"PDF file not found at path:\n\n{path}"
            }
        )
    return pdf

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
            "text": f"FORMATTED DOCUMENTS PREVIEW:\n\n{formatted_docs[0][:200]}"
        }
    )
    return formatted_docs

def retry(max_retries: int = 3, delay: float = 1.0):
    """
    A simple retry decorator that retries a function call upon encountering an exception.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    log_message(
                        {
                            "type":"ERROR",
                            "text": f"Error in {func.__name__}: {e}. Attempt {attempts} of {max_retries}"
                        }
                    )
                    if attempts >= max_retries:
                        log_message(
                            {
                                "type":"ERROR",
                                "text": f"Max retries reached for {func.__name__}. Raising exception."
                            }
                        )
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator