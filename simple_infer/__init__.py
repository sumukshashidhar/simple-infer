"""simple-infer: Minimal, hackable batch inference library for LLMs."""

from .inference import infer, call_llm

__version__ = "0.1.0"
__all__ = ["infer", "call_llm"]