import pytest
from unittest.mock import Mock, patch
from typing import List

# Import your classes
from syn_agent.llm_providers import BaseLLM, OpenAIModel, GoogleAIModel
from syn_agent.messages import Message

# Mock responses
mock_openai_response = Mock()
mock_openai_response.choices = [Mock(message="This is a mocked OpenAI response")]

mock_genai_response = Mock()
mock_genai_response.text = "This is a mocked Google AI response"

class TestOpenAIModel:
    @patch('openai.OpenAI')
    def test_init(self, mock_openai):
        """Test initialization of OpenAI model"""
        model = OpenAIModel(
            model="gpt-4o-2024-05-13", 
            api_key="fake-key", 
            max_tokens=100, 
            temperature=0.7
        )
        assert model.model == "gpt-4o-2024-05-13"
        assert model.max_tokens == 100
        assert model.temperature == 0.7
        mock_openai.assert_called_once_with(api_key="fake-key")

    @patch.object(OpenAIModel, 'client')
    def test_generate(self, mock_client):
        """Test generate method"""
        model = OpenAIModel(api_key="fake-key")
        mock_client.chat.completions.create.return_value = mock_openai_response
        
        result = model.generate("Test prompt")
        mock_client.chat.completions.create.assert_called_once()
        assert result == mock_openai_response.choices[0].message

    @patch.object(OpenAIModel, 'client')
    def test_chat(self, mock_client):
        """Test chat method"""
        model = OpenAIModel(api_key="fake-key")
        mock_client.chat.completions.create.return_value = mock_openai_response
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        
        result = model.chat(messages)
        mock_client.chat.completions.create.assert_called_once()
        assert result == mock_openai_response.choices[0].message

    @patch.object(OpenAIModel, 'generate')
    @patch.object(OpenAIModel, 'chat')
    def test_call(self, mock_chat, mock_generate):
        """Test __call__ method"""
        model = OpenAIModel(api_key="fake-key")
        mock_generate.return_value = "Generated text"
        mock_chat.return_value = "Chat response"
        
        messages = [{"role": "user", "content": "Hello"}]
        
        # Fix: Should pass input string to generate method
        model("Test prompt")
        mock_generate.assert_called_once_with(prompt="Test prompt")
        
        model(messages)
        mock_chat.assert_called_once_with(messages=messages)


class TestGoogleAIModel:
    @patch('google.genai.Client')
    def test_init(self, mock_client):
        """Test initialization of Google AI model"""
        model = GoogleAIModel(
            model="gemini-2.0-flash", 
            api_key="fake-key", 
            max_tokens=200, 
            temperature=0.5
        )
        assert model.model == "gemini-2.0-flash"
        assert model.max_tokens == 200
        assert model.temperature == 0.5
        assert model.top_p == 0.5
        mock_client.assert_called_once_with(api_key="fake-key")

    @patch.object(GoogleAIModel, 'client')
    def test_generate(self, mock_client):
        """Test generate method"""
        model = GoogleAIModel(api_key="fake-key")
        mock_client.models.generate_content.return_value = mock_genai_response
        
        result = model.generate("Test prompt")
        mock_client.models.generate_content.assert_called_once()
        assert result == mock_genai_response.text

    @patch.object(GoogleAIModel, 'client')
    def test_chat(self, mock_client):
        """Test chat method with system instruction"""
        model = GoogleAIModel(api_key="fake-key")
        mock_chat = Mock()
        mock_chat.send_message.return_value = mock_genai_response
        mock_client.chats.create.return_value = mock_chat
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ]
        
        result = model.chat(messages)
        mock_client.chats.create.assert_called_once()
        mock_chat.send_message.assert_called_once_with("Hello")
        assert result == mock_genai_response.text

    @patch.object(GoogleAIModel, 'generate')
    @patch.object(GoogleAIModel, 'chat')
    def test_call(self, mock_chat, mock_generate):
        """Test __call__ method"""
        model = GoogleAIModel(api_key="fake-key")
        mock_generate.return_value = "Generated text"
        mock_chat.return_value = "Chat response"
        
        messages = [{"role": "user", "content": "Hello"}]
        
        # Fix: Should pass input string to generate method
        model("Test prompt")
        mock_generate.assert_called_once_with(prompt="Test prompt")
        
        model(messages)
        mock_chat.assert_called_once_with(messages=messages)