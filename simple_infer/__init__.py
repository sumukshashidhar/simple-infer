"""simple-infer: Minimal, hackable batch inference library for LLMs."""

from .inference import batch_infer_conversations, call_llm

__version__ = "0.1.0"
__all__ = ["batch_infer_conversations", "call_llm"]