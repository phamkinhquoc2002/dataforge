from prompts import SFT, DPO, CONVERSATION, SYSTEM_PROMPT
from api_providers import Message
from tasks import Task

def key_map(dictionary: dict, key, default=None):
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
    system_prompt= {"role": "system", "content": SYSTEM_PROMPT}
    user_prompt = {"role": "user", "content": user_prompt_initialize(task)}
    message.append(system_prompt)
    message.append(user_prompt)
    return message
    