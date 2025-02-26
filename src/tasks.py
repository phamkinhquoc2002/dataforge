from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional

class Task(BaseModel):
    """
    Task that user wants agent to perform.
    """
    task_name: Literal['sft', 'dpo', 'multi-dialogue']
    localization: Optional[str] = Field(default=None, max_length=200, description="localize the part where you want to extract knowledge!")
    grounded_knowledge: Optional[str] = Field(default=None, max_length=200, description="grounded context to generate data from!")
    task_description: Optional[str]  = Field(default=None, max_length=200, description="additional task description tailor to personal needs!")
    num_of_data: int = Field(..., gt=3, lt=10)
    language: Optional[str]

    @field_validator('num_of_data')
    def check_num_of_data(cls, v: int):
        if v < 3 or v > 10:
            raise ValueError('You can only generate between 3 and 10 data samples for each batch')
        return v
    
    @field_validator('task_description', 'localization')
    def check_task(cls, v: str):
        if len(v) > 200:
            raise ValueError('Your task description should not exceed 200 characters')
        return v