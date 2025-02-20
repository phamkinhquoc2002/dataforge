from pydantic import BaseModel, Field, field_validator
from typing import Literal

class Task(BaseModel):
    task_name: Literal['sft', 'dpo', 'multi-dialogue']
    grounded_knowledge: str | None = None
    task_description: str | None  = Field(default=None, max_length=200)
    num_of_data: int = Field(..., gt=3, lt=10)
    language: str | None

    @field_validator('num_of_data')
    def check_num_of_data(cls, v: int):
        if v < 3 or v > 10:
            raise ValueError('You can only generate between 3 and 10 data samples for each batch')
        return v
    
    @field_validator('task_description')
    def check_task(cls, v: str):
        if len(v) > 200:
            raise ValueError('Your task description should not exceed 200 characters')
        return v
