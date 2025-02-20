from pydantic import BaseModel,Field, field_validator
from typing import Literal

class Task(BaseModel):
    task_name: Literal['sft', 'dpo', 'multi-dialogue']
    task_description: str 
    task_format: str
    num_of_data: int = Field(..., gt=3, lt=10)
    grounded_knowledge: str | None = None
    language: str | None

    @field_validator('num_of_data')
    def check_num_of_data(cls, v: int):
        if v < 3 or v > 10:
            raise ValueError('num_of_data should be greater than 3 and less than 10')
        return v




