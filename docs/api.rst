API Reference
=============

Inference Functions
-------------------

.. automodule:: simple_infer.inference
   :members: batch_infer_conversations, batch_infer_job, call_llm
   :undoc-members:
   :show-inheritance:

Pydantic Models
---------------

.. automodule:: simple_infer.models
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

Basic Usage (Dict-based)
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from simple_infer import batch_infer_conversations
   
   conversations = [
       [{"role": "user", "content": "What is 2+2?"}],
       [{"role": "user", "content": "What is 3+3?"}]
   ]
   
   results = batch_infer_conversations(conversations, model="gpt-4o-mini")
   print(results)  # ['4', '6']

Pydantic Model Usage
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from simple_infer import batch_infer_job, InferenceJob, Conversation, Message
   
   # Create structured job with Pydantic models
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
   print(f"Success: {result.success_count}, Failed: {result.failure_count}")
   print(result.responses)

Async Usage
~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from openai import AsyncOpenAI
   from simple_infer import call_llm
   
   async def main():
       client = AsyncOpenAI()
       messages = [{"role": "user", "content": "Hello!"}]
       result = await call_llm(client, messages, model="gpt-4o-mini")
       print(result)
   
   asyncio.run(main())

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from simple_infer import InferenceJob, Conversation, Message, batch_infer_job
   
   # Complex job with custom configuration
   job = InferenceJob(
       conversations=[
           Conversation(messages=[
               Message(role="system", content="You are a helpful assistant."),
               Message(role="user", content="Explain quantum computing in one sentence.")
           ])
       ],
       model="gpt-4o-mini",
       base_url="https://api.openai.com/v1",  # or custom endpoint
       max_concurrent=5,
       temperature=0.3,
       max_tokens=100
   )
   
   result = batch_infer_job(job)
   print(result.responses[0])