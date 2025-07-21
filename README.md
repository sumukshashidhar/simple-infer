# simple-infer

Minimal, hackable batch inference library for LLMs. No batch endpoints needed.

## Installation

```bash
pip install simple-infer
```

## Usage

```python
from simple_infer import infer

conversations = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]
]

results = infer(conversations, model="gpt-4.1-nano", max_concurrent=32)
```

## Features

- **Simple**: Two main functions - `infer()` and `call_llm()`
- **Fast**: Async batch processing with configurable concurrency
- **Reliable**: Built-in retries with exponential backoff
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
