"""
Synthetic Data Generation Agentic System.

A framework for generating synthetic data using Google AI models with human-in-the-loop feedback.
"""

from .agent import Agent
from .multi_agent import SyntheticDataGenerator, Feedback, CurrentState
from .llm_providers import GoogleAIModel
from .messages import Message
from .tasks import Task
from .utils import pdf_parser, prompt_initialize, one_shot_prompt, extract_valid_output, save_to_file, log_message, document_format

__version__ = "0.1.0"
__all__ = [
    "Agent",
    "SyntheticDataGenerator",
    "Feedback",
    "CurrentState",
    "GoogleAIModel",
    "Message",
    "Task",
    "pdf_parser",
    "prompt_initialize",
    "one_shot_prompt",
    "extract_valid_output",
    "save_to_file",
    "log_message",
    "document_format"
]
