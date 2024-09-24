Language Metrics
================

The Language Metrics module provides a suite of tools for evaluating the quality and consistency of language model outputs. 
These metrics are designed to assess various aspects of generated text, such as semantic similarity, factual accuracy, and coherence. 
By leveraging advanced models and techniques, the Language Metrics module helps ensure that language models produce high-quality, reliable, and contextually appropriate responses.

The module includes the following key metrics:

- **BLEURTScore**: A learned metric that uses BERT to compute a similarity score for each token in the candidate sentence with each token in the reference sentence. This metric is particularly useful for evaluating the semantic content of generated text.
- **Q-Squared**: A reference-free metric that evaluates the factual consistency of knowledge-grounded dialogue systems. This approach is based on automatic question generation and question answering, providing a robust measure of how well the generated responses align with the given knowledge.

These metrics are essential for developers and researchers working on natural language processing (NLP) tasks, as they provide valuable insights into the performance and reliability of language models.

.. automodule:: saga_llm_evaluation.helpers.language_metrics
   :members:
   :undoc-members:
   :show-inheritance: