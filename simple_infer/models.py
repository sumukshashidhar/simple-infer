"""Pydantic models for job submission and API interaction."""

from pydantic import BaseModel, Field
from typing import Literal


class Message(BaseModel):
    """A single message in a conversation."""
    role: Literal["system", "user", "assistant"]
    content: str


class Conversation(BaseModel):
    """A conversation consisting of multiple messages."""
    messages: list[Message]
    
    def to_dict_list(self) -> list[dict]:
        """Convert to list of dicts format expected by OpenAI API."""
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]


class InferenceJob(BaseModel):
    """A batch inference job configuration."""
    conversations: list[Conversation]
    model: str = Field(default="gpt-4o-mini", description="Model to use for inference")
    base_url: str = Field(default="https://api.openai.com/v1", description="API base URL")
    max_concurrent: int = Field(default=64, description="Maximum concurrent requests")
    temperature: float | None = Field(default=None, description="Sampling temperature")
    max_tokens: int | None = Field(default=None, description="Maximum tokens to generate")
    
    def to_conversations_list(self) -> list[list[dict]]:
        """Convert to the format expected by batch_infer_conversations."""
        return [conv.to_dict_list() for conv in self.conversations]
    
    def get_api_kwargs(self) -> dict:
        """Get API parameters as a dict for passing to inference functions."""
        kwargs = {
            "model": self.model,
            "base_url": self.base_url, 
            "max_concurrent": self.max_concurrent
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        return kwargs


class InferenceResult(BaseModel):
    """Result of a batch inference job."""
    responses: list[str]
    job_config: InferenceJob
    success_count: int
    failure_count: int
    
    @classmethod
    def from_responses(cls, responses: list[str], job: InferenceJob) -> "InferenceResult":
        """Create result from raw responses and job config."""
        success_count = sum(1 for r in responses if r.strip())
        failure_count = len(responses) - success_count
        return cls(
            responses=responses,
            job_config=job,
            success_count=success_count,
            failure_count=failure_count
        )