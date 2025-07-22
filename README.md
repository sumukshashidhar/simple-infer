# simple-infer

Minimal, hackable batch inference library for LLMs. No batch endpoints needed.

## Installation

```bash
pip install simple-infer
```

## Quick Start

### Dict-based API (Simple)

```python
from simple_infer import batch_infer_conversations

conversations = [
    [{"role": "user", "content": "What is 2+2?"}],
    [{"role": "user", "content": "What is 3+3?"}]
]

results = batch_infer_conversations(conversations, model="gpt-4o-mini")
print(results)  # ['4', '6']
```

### Pydantic API (Structured)

```python
from simple_infer import batch_infer_job, InferenceJob, Conversation, Message

job = InferenceJob(
    conversations=[
        Conversation(messages=[
            Message(role="user", content="What is 2+2?")
        ]),
        Conversation(messages=[
            Message(role="user", content="What is 3+3?")
        ])
    ],
    model="gpt-4o-mini",
    max_concurrent=10,
    temperature=0.7
)

result = batch_infer_job(job)
print(f"Successful: {result.success_count}, Failed: {result.failure_count}")
print(result.responses)
```

## Features

- **Simple**: Two APIs - dict-based for quick scripts, Pydantic for structured use
- **Fast**: Async batch processing with configurable concurrency
- **Reliable**: Built-in retries with exponential backoff  
- **Typed**: Full Pydantic model support with validation
- **Hackable**: Clean, readable code you can modify

## Development

### Setup
```bash
# Clone and install dependencies
uv sync --extra test --extra docs

# Copy environment template and add your OpenAI API key
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=your-key-here
```

### Testing
```bash
# Run tests (requires OPENAI_API_KEY in environment)
export OPENAI_API_KEY=your-key-here
uv run pytest tests/ -v

# Or use .env file
uv run pytest tests/ -v
```
