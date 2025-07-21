.. simple-infer documentation master file, created by
   sphinx-quickstart on Mon Jul 21 14:19:45 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

simple-infer
============

Minimal, hackable batch inference library for LLMs. No batch endpoints needed.

Installation
------------

.. code-block:: bash

   pip install simple-infer

Quick Start
-----------

.. code-block:: python

   from simple_infer import infer

   conversations = [
       [{"role": "user", "content": "What is 2+2?"}],
       [{"role": "user", "content": "What is the capital of France?"}]
   ]

   results = infer(conversations, model="gpt-4o-mini")
   print(results)

Features
--------

- **Simple**: Two main functions - ``infer()`` and ``call_llm()``
- **Fast**: Async batch processing with configurable concurrency  
- **Reliable**: Built-in retries with exponential backoff
- **Hackable**: Clean, readable code you can modify

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   api

