import pytest
from unittest.mock import Mock, patch
from src.multi_agent import SyntheticDataGenerator, CurrentState
from src.tasks import Task
from src.llm_providers import GoogleAIModel, BaseLLM
from src.messages import Message
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

@pytest.fixture
def mock_google_ai():
    llm = Mock(spec=GoogleAIModel)
    llm.return_value = "Mocked Google AI response"
    return llm

@pytest.fixture
def mock_retriever():
    retriever = Mock(spec=BM25Retriever)
    retriever.invoke.return_value = [
        Document(page_content="Test document 1"),
        Document(page_content="Test document 2")
    ]
    return retriever

@pytest.fixture
def example_task():
    return Task(
        task_name="dpo",
        localization='Optimization, Data Science and AI',
        task_description="Generate a conversation between a HR manager and a candidate",
        rows_per_batch=15,
        batch_size=20,
        language="English"
    )

@pytest.fixture
def valid_task():
    return Task(
        task_name="sft",
        localization="test",
        task_description="Generate synthetic data for testing",
        rows_per_batch=15,
        batch_size=20,
        language="English"
    )

@pytest.fixture
def mock_llm():
    mock = Mock(spec=BaseLLM)
    mock.generate.return_value = "Test response"
    return mock

@pytest.mark.asyncio
async def test_initialization(mock_google_ai, mock_retriever):
    """Test that the SyntheticDataGenerator initializes correctly"""
    agent = SyntheticDataGenerator(
        llm=mock_google_ai,
        retriever=mock_retriever,
        output_path="./test_output",
        buffer_size=5,
        thread_id={"configurable": {"thread_id": 123}}
    )
    
    assert agent.llm == mock_google_ai
    assert agent.retriever == mock_retriever
    assert agent.buffer_size == 5
    assert agent.output_path == "./test_output"

@pytest.mark.asyncio
async def test_retrieve_documents(mock_google_ai, mock_retriever, example_task):
    """Test document retrieval functionality"""
    agent = SyntheticDataGenerator(
        llm=mock_google_ai,
        retriever=mock_retriever,
        output_path="./test_output",
        thread_id={"configurable": {"thread_id": 123}}
    )
    
    initial_state = CurrentState(
        task=example_task,
        conversations=[],
        retrieved_documents=None,
        response=None,
        human_feedback=None
    )
    
    command = agent.retrieve(initial_state)
    assert command.goto == 'fish_for_feedback'
    assert command.update['retrieved_documents'] is not None

@pytest.mark.asyncio
async def test_human_feedback_loop(mock_google_ai, mock_retriever, example_task):
    """Test the human feedback loop functionality"""
    agent = SyntheticDataGenerator(
        llm=mock_google_ai,
        retriever=mock_retriever,
        output_path="./test_output",
        thread_id={"configurable": {"thread_id": 123}}
    )
    
    # Mock the interrupt function
    with patch('src.multi_agent.interrupt', return_value="yes"):
        initial_state = CurrentState(
            task=example_task,
            conversations=[
                Message(role="user", content="Initial prompt"),
                Message(role="assistant", content="Initial response")
            ],
            retrieved_documents=["Test document"],
            response="Test response",
            human_feedback=None
        )
        
        command = agent.approve(initial_state)
        assert command.goto == 'data_generate'

def test_retrieve_documents(valid_task, mock_llm, mock_retriever):
    """Test document retrieval."""
    generator = SyntheticDataGenerator(
        llm=mock_llm,
        retriever=mock_retriever,
        output_path="./test_output",
        thread_id={"configurable": {"thread_id": 123}}
    )
    state = CurrentState(
        task=valid_task,
        conversations=[],
        retrieved_documents=None,
        response=None,
        human_feedback=None
    )
    command = generator.retrieve(state)
    assert command.goto == 'fish_for_feedback'
    assert command.update['retrieved_documents'] is not None

def test_human_feedback_loop(valid_task, mock_llm, mock_retriever):
    """Test human feedback loop."""
    generator = SyntheticDataGenerator(
        llm=mock_llm,
        retriever=mock_retriever,
        output_path="./test_output",
        thread_id={"configurable": {"thread_id": 123}}
    )
    state = CurrentState(
        task=valid_task,
        conversations=[Message(role="user", content="Initial message")],
        retrieved_documents=["Test document"],
        response="Test response",
        human_feedback=None
    )
    with patch('src.multi_agent.interrupt', return_value="yes"):
        command = generator.approve(state)
        assert command.goto == 'data_generate'