"""Tests for simple-infer using gpt-4.1-nano."""
import asyncio
import os
import pytest
from openai import AsyncOpenAI
from simple_infer import infer, call_llm


@pytest.fixture
def openai_key():
    """Get OpenAI API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return api_key


@pytest.fixture
def simple_conversations():
    """Simple test conversations."""
    return [
        [
            {"role": "system", "content": "You are a helpful assistant. Respond in one word."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant. Respond in one word."},
            {"role": "user", "content": "What is 2+2?"},
        ]
    ]


def test_sync_infer(openai_key, simple_conversations):
    """Test synchronous inference with gpt-4.1-nano."""
    os.environ["OPENAI_API_KEY"] = openai_key
    
    results = infer(
        simple_conversations,
        base_url="https://api.openai.com/v1",
        model="gpt-4.1-nano",
        max_concurrent=2
    )
    
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)
    assert all(len(r.strip()) > 0 for r in results)


@pytest.mark.asyncio
async def test_async_call_llm(openai_key):
    """Test async call_llm function directly."""
    client = AsyncOpenAI(api_key=openai_key)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Respond in one word."},
        {"role": "user", "content": "What is the capital of Japan?"}
    ]
    
    result = await call_llm(client, messages, model="gpt-4.1-nano")
    
    assert isinstance(result, str)
    assert len(result.strip()) > 0


@pytest.mark.asyncio
async def test_batch_processing(openai_key, simple_conversations):
    """Test that batch processing works correctly."""
    from simple_infer.inference import _async_batch_infer
    
    os.environ["OPENAI_API_KEY"] = openai_key
    
    results = await _async_batch_infer(
        simple_conversations,
        base_url="https://api.openai.com/v1",
        model="gpt-4.1-nano",
        max_concurrent=2
    )
    
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)


def test_empty_conversations(openai_key):
    """Test handling of empty conversation list."""
    os.environ["OPENAI_API_KEY"] = openai_key
    
    results = infer([], model="gpt-4.1-nano")
    
    assert results == []


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling with invalid messages."""
    client = AsyncOpenAI(api_key="invalid-key")
    
    messages = [{"role": "user", "content": "Hello"}]
    
    # Should return empty string on error after retries
    result = await call_llm(client, messages, model="gpt-4.1-nano")
    
    assert result == ""