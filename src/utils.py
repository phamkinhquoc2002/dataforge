from prompts import SFT, DPO, CONVERSATION, SYSTEM_PROMPT
from src.messages import Message
from tasks import Task
from typing import List
from pypdf import PdfReader
import re
import json
import os
import logging
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('__name__')

def key_map(dictionary: dict, key, default=None) -> str:
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

def user_prompt_initialize(task: Task) -> str:
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
    num_of_data = task.rows_per_batch
    if task.task_description:
        task_description = f"Additional Dataset Info: {task.task_description}"
    else:
        task_description = ""
    if task.language:
        language = f" entirely in {task.language}"
    else:
        language = ""
    return f"You are tasked to help me generate a dataset of {num_of_data} rows{language}, based entirely on the following context:\n{grounded}\n{task_format}\n{task_description}\n"
       
def prompt_initialize(
        task: Task
) -> List[Message]:
    """
    Initialize a conversation with the synthetic data generator.
    
    Returns:
        A message containing the system prompt.
    """
    messages = []
    system_prompt= {"role": "system", 
                    "content": SYSTEM_PROMPT}
    user_prompt = {"role": "user", 
                   "content": user_prompt_initialize(task)}
    messages.append(system_prompt)
    messages.append(user_prompt)
    return messages

def prompt_validator(func):
    """
    Validate the prompt format for data generator pipeline.
    """
    def wrapper(*args, **kwargs):
        logger.info(f"Calling {func.__name__}")
        docs = func(*args, **kwargs)
        assert docs[0]["role"] == "system", "Wrong message role"
        assert docs[1]["role"] == "user", "Wrong message role"
        logger.info(f"""System Prompt: {docs[0]['content'][:100]}\nUser Prompt: {docs[1]['content'][:100]}\n""")
        return docs
    return wrapper

@prompt_validator
def one_shot_prompt(user_prompt:Message, response: str) -> List[Message]:
    """
    Create a behavioral one-shot prompt to adapt the llm to user's preferred answer.
    """
    optimized_prompt = user_prompt[1]['content'] + f"""Example:\n{response}"""
    return optimized_prompt
    
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
    except:
        error = "The format is invalid to extract. You must return the exact format as specified in the prompt"
        logger.error(error)   

def save_to_file(output: List[dict], filename: str) -> None:
    """
    Save the output to a file.
    
    Parameters:
        output (List[dict]): The output to save.
        filename (str): The name of the file to save to.
    """
    try:
        with open(filename, 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = []
    existing_data.extend(output)
    with open(filename, 'w') as f:
        json.dump(existing_data, f)

def chunk_parser(func):
    """
    Decorator to split PDF content into smaller, manageable chunks.
    """
    def wrapper(*args, **kwargs):
        logger.info(f"Calling {func.__name__}")
        docs = func(*args, **kwargs)
        documents = [Document(page_content=doc) for doc in docs]
        logger.info(f"Loading {len(documents)} pieces of context!")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4096,
            chunk_overlap=20
        )
        splitted_texts = text_splitter.split_documents(documents)
        
        if splitted_texts:
            preview = splitted_texts[0].page_content[:100]
            logger.info(f"Split successful. First chunk preview: {preview}...")
        return splitted_texts
    return wrapper

@chunk_parser
def pdf_parser(path: str):
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
                logger.info(f"Extracted {len(documents)} pages. First page preview: {documents[0][:100]}")
            else:
                logger.warning("PDF file contains no extractable text")
        except Exception as e:
            logger.error(f"Error parsing PDF: {str(e)}")
    else:
        logger.error(f"PDF file not found at path: {path}")
    return documents