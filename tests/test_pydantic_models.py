"""Tests for Pydantic models and batch_infer_job function."""

import pytest
from simple_infer.models import Message, Conversation, InferenceJob, InferenceResult


def test_message_creation():
    """Test Message model creation and validation."""
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_message_invalid_role():
    """Test Message validation with invalid role."""
    with pytest.raises(ValueError):
        Message(role="invalid", content="Hello")


def test_conversation_creation():
    """Test Conversation model creation."""
    conv = Conversation(messages=[
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!")
    ])
    assert len(conv.messages) == 2
    assert conv.messages[0].role == "user"


def test_conversation_to_dict_list():
    """Test conversation conversion to dict format."""
    conv = Conversation(messages=[
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi!")
    ])
    
    dict_list = conv.to_dict_list()
    expected = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"}
    ]
    assert dict_list == expected


def test_inference_job_creation():
    """Test InferenceJob model creation."""
    job = InferenceJob(
        conversations=[
            Conversation(messages=[
                Message(role="user", content="Test")
            ])
        ],
        model="gpt-4o-mini",
        max_concurrent=5
    )
    
    assert len(job.conversations) == 1
    assert job.model == "gpt-4o-mini"
    assert job.max_concurrent == 5


def test_inference_job_defaults():
    """Test InferenceJob with default values."""
    job = InferenceJob(
        conversations=[
            Conversation(messages=[
                Message(role="user", content="Test")
            ])
        ]
    )
    
    assert job.model == "gpt-4o-mini"
    assert job.base_url == "https://api.openai.com/v1"
    assert job.max_concurrent == 64


def test_inference_job_to_conversations_list():
    """Test InferenceJob conversion to conversations list."""
    job = InferenceJob(
        conversations=[
            Conversation(messages=[
                Message(role="user", content="Hello")
            ]),
            Conversation(messages=[
                Message(role="user", content="World")
            ])
        ]
    )
    
    convos = job.to_conversations_list()
    expected = [
        [{"role": "user", "content": "Hello"}],
        [{"role": "user", "content": "World"}]
    ]
    assert convos == expected


def test_inference_job_get_api_kwargs():
    """Test InferenceJob API kwargs generation."""
    job = InferenceJob(
        conversations=[
            Conversation(messages=[Message(role="user", content="Test")])
        ],
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=100
    )
    
    kwargs = job.get_api_kwargs()
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_tokens"] == 100
    assert kwargs["max_concurrent"] == 64


def test_inference_result_creation():
    """Test InferenceResult model creation."""
    job = InferenceJob(
        conversations=[
            Conversation(messages=[Message(role="user", content="Test")])
        ]
    )
    
    responses = ["Response 1", "", "Response 3"]
    result = InferenceResult.from_responses(responses, job)
    
    assert result.responses == responses
    assert result.job_config == job
    assert result.success_count == 2  # Non-empty responses
    assert result.failure_count == 1   # Empty response