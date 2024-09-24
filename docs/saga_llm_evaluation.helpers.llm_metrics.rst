LLM Metrics
===========

The idea is to use the LLMs to score the candidate output based on its generation probability without any reference target, under the assumption that the LLMs have learned to assign higher probabilities to high-quality and fluent texts.
Obviously, the disadvantage of this approach is that the LLMs are not perfect and can assign high probabilities to incorrect or nonsensical texts.
However, the advantage is that it is a fully automatic and reference-free evaluation method that can be used to evaluate the quality of the output of the LLMs in a large-scale setting.

.. automodule:: saga_llm_evaluation.helpers.llm_metrics
   :members:
   :undoc-members:
   :show-inheritance: