API Reference
=============

Main Functions
--------------

.. automodule:: simple_infer.inference
   :members: infer, call_llm
   :undoc-members:
   :show-inheritance:

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from simple_infer import infer
   
   conversations = [
       [{"role": "user", "content": "What is 2+2?"}],
       [{"role": "user", "content": "What is 3+3?"}]
   ]
   
   results = infer(conversations, model="gpt-4.1-nano")
   print(results)  # ['4', '6']

Async Usage
~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from openai import AsyncOpenAI
   from simple_infer import call_llm
   
   async def main():
       client = AsyncOpenAI()
       messages = [{"role": "user", "content": "Hello!"}]
       result = await call_llm(client, messages, model="gpt-4.1-nano")
       print(result)
   
   asyncio.run(main())

Batch Processing with Concurrency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from simple_infer import infer
   
   # Process 100 conversations with max 32 concurrent requests
   conversations = [
       [{"role": "user", "content": f"Count to {i}"}] 
       for i in range(1, 101)
   ]
   
   results = infer(
       conversations,
       model="gpt-4.1-nano",
       max_concurrent=32,
       temperature=0.7
   )