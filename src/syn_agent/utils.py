from prompts import SFT, DPO, CONVERSATION, SYSTEM_PROMPT
from messages import Message
from tasks import Task
from typing import List
import re
import json
import os

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
    num_of_data = task.num_of_data
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
) -> Message:
    """
    Initialize a conversation with the synthetic data generator.
    
    Returns:
        A message containing the system prompt.
    """
    message = []
    system_prompt= {"role": "system", 
                    "content": SYSTEM_PROMPT}
    user_prompt = {"role": "user", 
                   "content": user_prompt_initialize(task)}
    message.append(system_prompt)
    message.append(user_prompt)
    return message
    
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
        error = "The format is invalid to extract. You must return the exact format as specified in the prompt"
        return error
    
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
        
        
    