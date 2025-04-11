import pytest
from unittest.mock import Mock, patch
from src.llm_providers import GoogleAIModel
from src.messages import Message
from google.genai import types

@pytest.fixture
def mock_google_ai_response():
    mock_response = Mock()
    mock_response.text = "Test response"
    return mock_response

@pytest.fixture
def mock_google_ai_chat():
    mock_chat = Mock()
    mock_chat.send_message.return_value = Mock(text="Test chat response")
    return mock_chat

def test_google_ai_initialization():
    """Test Google AI model initialization."""
    model = GoogleAIModel(
        model="gemini-2.0-flash",
        api_key="test_key",
        max_tokens=1000,
        temperature=0.7
    )
    
    assert model.model == "gemini-2.0-flash"
    assert model.api_key == "test_key"
    assert model.max_tokens == 1000
    assert model.temperature == 0.7
    assert isinstance(model.config, types.GenerateContentConfig)

@patch('google.genai.Client')
def test_google_ai_generate(mock_client, mock_google_ai_response):
    """Test Google AI model generate method."""
    # Setup mock
    mock_client.return_value.models.generate_content.return_value = mock_google_ai_response
    
    model = GoogleAIModel(api_key="test_key")
    response = model.generate("Test prompt")
    
    assert response == "Test response"
    mock_client.return_value.models.generate_content.assert_called_once()

@patch('google.genai.Client')
def test_google_ai_chat(mock_client, mock_google_ai_chat):
    """Test Google AI model chat method."""
    # Setup mock
    mock_client.return_value.chats.create.return_value = mock_google_ai_chat
    
    model = GoogleAIModel(api_key="test_key")
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there"),
        Message(role="user", content="How are you?")
    ]
    
    response = model.chat(messages)
    assert response == "Test chat response"
    mock_client.return_value.chats.create.assert_called_once()

@patch('google.genai.Client')
def test_google_ai_error_handling(mock_client):
    """Test Google AI model error handling."""
    # Setup mock to raise exception
    mock_client.return_value.models.generate_content.side_effect = Exception("API Error")
    
    model = GoogleAIModel(api_key="test_key")
    
    with pytest.raises(Exception):
        model.generate("Test prompt")

def test_google_ai_config():
    """Test Google AI model configuration."""
    model = GoogleAIModel(
        api_key="test_key",
        max_tokens=2000,
        temperature=0.8,
        top_p=0.7
    )
    
    config = model.config
    assert config.max_output_tokens == 2000
    assert config.temperature == 0.8
    
    # Test config update
    new_config = types.GenerateContentConfig(
        max_output_tokens=3000,
        temperature=0.9
    )
    model.config = new_config
    assert model.config.max_output_tokens == 3000
    assert model.config.temperature == 0.9

@patch('google.genai.Client')
def test_google_ai_call_method(mock_client):
    """Test Google AI model __call__ method."""
    # Setup mock response
    mock_response = Mock()
    mock_response.text = "generated response"
    mock_client.return_value.models.generate_content.return_value = mock_response
    
    model = GoogleAIModel(api_key="test_key")
    
    # Test with string input
    result = model("test prompt")
    assert result == "generated response"
    mock_client.return_value.models.generate_content.assert_called_once()
    
    # Setup mock chat
    mock_chat = Mock()
    mock_chat.send_message.return_value = Mock(text="chat response")
    mock_client.return_value.chats.create.return_value = mock_chat
    
    # Test with message list input
    messages = [Message(role="user", content="test")]
    result = model(messages)
    assert result == "chat response"
    mock_client.return_value.chats.create.assert_called_once()
    
    # Test with invalid input
    with pytest.raises(ValueError):
        model(123)  # Invalid input type 