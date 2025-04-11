import pytest
from src.multi_agent import Feedback, CurrentState, Message, Task
from typing import List, Dict, Any

@pytest.fixture
def mock_messages():
    return [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!")
    ]

@pytest.fixture
def mock_feedback():
    return Feedback(
        feedback="Good response",
    )

@pytest.fixture
def mock_state():
    task = Task(
        task_name="sft",
        localization="test",
        task_description="Test task",
        rows_per_batch=15,
        batch_size=20,
        language="English"
    )
    return CurrentState(
        task=task,
        human_feedback="good",
        response="Response from the agent",
        conversations=[
        Message(role="user", content="Hello")],
        retrieved_documents=["Document1", "Document2"]
    )


def test_feedback_validation():
    """Test Feedback validation."""
    with pytest.raises(ValueError):
        Feedback(feedback="a" * 301)  # Exceeds 300 characters

def test_state_creation():
    """Test CurrentState object creation."""
    task = Task(
        task_name="sft",
        localization="test",
        task_description="Test task",
        rows_per_batch=15,
        batch_size=20,
        language="English"
    )
    state = CurrentState(
        task=task,
        conversations=[Message(role="user", content="Hello")],
        human_feedback=None,
        response=None,
        retrieved_documents=None
    )
    assert len(state["conversations"]) == 1
    assert state["task"] == task

def test_state_message_append():
    """Test appending messages to CurrentState."""
    task = Task(
        task_name="sft",
        localization="test",
        task_description="Test task",
        rows_per_batch=15,
        batch_size=20,
        language="English"
    )
    state = CurrentState(
        task=task,
        conversations=[Message(role="user", content="Hello")],
        human_feedback=None,
        response=None,
        retrieved_documents=None
    )
    new_message = Message(role="assistant", content="Hi there")
    state["conversations"].append(new_message)
    assert len(state["conversations"]) == 2
    assert state["conversations"][1].content == "Hi there"

def test_state_message_append(mock_state):
    new_message = Message(role="user", content="New message")
    mock_state["conversations"].append(new_message)
    assert len(mock_state["conversations"]) == 2
    assert mock_state["conversations"][-1] == new_message 