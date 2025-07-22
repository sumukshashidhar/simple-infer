"""
Performant, Batch Inference File, w/o the batch endpoint.

Exposes functions for both raw dict-based and Pydantic model-based inference.
"""
import asyncio
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
import httpx
from tqdm.asyncio import tqdm_asyncio
from .models import InferenceJob, InferenceResult

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def _call_llm_with_retries(client: AsyncOpenAI, messages: list[dict], **kwargs) -> str:
    """Internal function that actually makes the API call - tenacity will retry this."""
    r = await client.chat.completions.create(messages=messages, **kwargs)
    return r.choices[0].message.content or ""

async def call_llm(client: AsyncOpenAI, messages: list[dict], **kwargs) -> str:
    """Call LLM with a single conversation.
    
    Args:
        client: AsyncOpenAI client instance
        messages: List of message dicts with 'role' and 'content' keys
        **kwargs: Additional parameters passed to OpenAI API (model, temperature, etc.)
        
    Returns:
        String response from the LLM, or empty string if all retries failed
        
    Example:
        >>> client = AsyncOpenAI()
        >>> messages = [{"role": "user", "content": "Hello"}]
        >>> result = await call_llm(client, messages, model="gpt-4.1-nano")
    """
    try:
        return await _call_llm_with_retries(client, messages, **kwargs)
    except Exception as e:
        logger.warning(f"API call failed after all retries: {e}")
        return ""

async def _batch_infer(client: AsyncOpenAI, convos: list[list[dict]], max_concurrent: int = 64, **kwargs) -> list[str]:
    sem = asyncio.Semaphore(max_concurrent)

    async def process(msgs):
        async with sem:
            return await call_llm(client, msgs, **kwargs)
    
    tasks = [process(msgs) for msgs in convos]
    return await tqdm_asyncio.gather(*tasks, desc="LLM calls")

async def _async_batch_infer(convos: list[list[dict]], base_url: str = "https://api.openai.com/v1", **kwargs) -> list[str]:
    """Async function that creates and manages the HTTP client properly."""
    max_conn = kwargs.get('max_concurrent', 64)
    
    # Create HTTP client in the same async context
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_conn),
        timeout=httpx.Timeout(60.0)
    ) as http_client:
        # Create OpenAI client with our HTTP client
        client = AsyncOpenAI(base_url=base_url, http_client=http_client)
        
        # Do the actual work
        return await _batch_infer(client, convos, **kwargs)

def batch_infer_conversations(convos: list[list[dict]], base_url: str = "https://api.openai.com/v1", **kwargs) -> list[str]:
    """Batch inference on multiple conversations.
    
    Args:
        convos: List of conversations, where each conversation is a list of message dicts
        base_url: OpenAI API base URL (default: "https://api.openai.com/v1")
        **kwargs: Additional parameters:
            - model: Model name (e.g., "gpt-4.1-nano")
            - max_concurrent: Max concurrent requests (default: 64)
            - temperature, max_tokens, etc.: OpenAI API parameters
            
    Returns:
        List of string responses, one per conversation
        
    Example:
        >>> conversations = [
        ...     [{"role": "user", "content": "What is 2+2?"}],
        ...     [{"role": "user", "content": "What is 3+3?"}]
        ... ]
        >>> results = batch_infer_conversations(conversations, model="gpt-4.1-nano", max_concurrent=10)
        >>> print(results)  # ['4', '6']
    """
    return asyncio.run(_async_batch_infer(convos, base_url, **kwargs))


def batch_infer_job(job: InferenceJob) -> InferenceResult:
    """Batch inference using a Pydantic InferenceJob model.
    
    Args:
        job: InferenceJob instance containing conversations and configuration
        
    Returns:
        InferenceResult containing responses and metadata
        
    Example:
        >>> from simple_infer.models import InferenceJob, Conversation, Message
        >>> job = InferenceJob(
        ...     conversations=[
        ...         Conversation(messages=[
        ...             Message(role="user", content="What is 2+2?")
        ...         ])
        ...     ],
        ...     model="gpt-4o-mini",
        ...     max_concurrent=10
        ... )
        >>> result = batch_infer_job(job)
        >>> print(result.responses[0])  # '4'
    """
    convos = job.to_conversations_list()
    kwargs = job.get_api_kwargs()
    responses = batch_infer_conversations(convos, **kwargs)
    return InferenceResult.from_responses(responses, job)

if __name__ == "__main__":
    convos = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Italy?"},
        ]
    ]
    
    results = batch_infer_conversations(convos, max_concurrent=32, model="Qwen/Qwen3-32B-FP8")
    
    for i, r in enumerate(results):
        logger.info(f"[{i}] {r or 'FAILED'}")
