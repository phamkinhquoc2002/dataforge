from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional

class Task(BaseModel):
    """
    Task that user wants agent to perform.
    """
    task_name: Literal['sft', 'dpo', 'multi-dialogue']
    localization: Optional[str] = Field(default=None, max_length=200, description="localize the part where you want to extract knowledge!")
    grounded_knowledge: Optional[str] = Field(default=None, description="grounded context to generate data from!")
    task_description: Optional[str]  = Field(default=None, max_length=200, description="additional task description tailor to personal needs!")
    rows_per_batch: int = Field(..., gt=10, lt=20, description="number of data rows generated per batch")
    batch_size: int = Field(default=20, description="total number of batches")
    language: Optional[str]

    @field_validator('num_of_data', mode='before')
    def check_num_of_data(cls, v: int):
        if v < 10 or v > 20:
            raise ValueError('You can only generate between 3 and 10 data samples for each batch')
        return v
    
    @field_validator('task_description', 'localization',  mode='before')
    def check_task(cls, v: str):
        if len(v) > 200:
            raise ValueError('Your task description should not exceed 200 characters')
        return v