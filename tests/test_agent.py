import pytest
from unittest.mock import Mock, patch
from src.agent import Agent
from src.messages import Message
from src.llm_providers import BaseLLM

class TestAgent(Agent):
    def __init__(self, llm: BaseLLM):
        self.llm = llm  # Initialize llm before calling super()
        super().__init__(llm)
        self.logs = []

    def log(self, message: str):
        self.logs.append(message)

@pytest.fixture
def mock_llm():
    mock = Mock(spec=BaseLLM)
    mock.generate.return_value = "Test response"
    return mock

def test_agent_initialization(mock_llm):
    """Test agent initialization."""
    agent = TestAgent(mock_llm)
    assert agent.llm == mock_llm
    assert isinstance(agent.logs, list)
    assert len(agent.logs) == 0

def test_agent_logging(mock_llm):
    """Test agent logging functionality."""
    agent = TestAgent(mock_llm)
    test_message = "Test log message"
    agent.log(test_message)
    assert len(agent.logs) == 1
    assert agent.logs[0] == test_message

def test_agent_multiple_logs(mock_llm):
    """Test agent logging multiple messages."""
    agent = TestAgent(mock_llm)
    messages = ["First message", "Second message", "Third message"]
    for msg in messages:
        agent.log(msg)
    assert len(agent.logs) == 3
    assert agent.logs == messages 