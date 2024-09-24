# Retrieval Metrics

Retrieval metrics are essential for evaluating the efficiency and effectiveness of information retrieval systems.
These metrics help in assessing how well a system retrieves relevant information in response to a query, either by comparing the retrieved results to a ground truth or by measuring the relevance of the retrieved chunks to the query.

In the context of large language models (LLMs), retrieval metrics play a crucial role in determining the model’s ability to fetch and present relevant information from a vast corpus of data.
These metrics ensure that the model not only generates coherent and contextually appropriate responses but also retrieves information that is accurate and pertinent to the user’s query.

<a id="module-saga_llm_evaluation.helpers.retrieval_metrics"></a>

### *class* saga_llm_evaluation.helpers.retrieval_metrics.Accuracy(index, k=2)

Bases: `object`

Accuracy computes the accuracy of the retrieved information relatively to the query.
In other words, it validates that the retrieved information is correct.

* **Parameters:**
  * **index** (*llama_index.core.index.Index*) – Index containing the information to retrieve.
  * **k** (*int* *,* *optional*) – Number of similar items to retrieve. Defaults to 2.

#### compute(query: str, expected_ids: list) → dict

This method computes a series of metrics of the retrieved information relatively to the query.

* **Parameters:**
  * **query** (*str*) – Query topic.
  * **expected_ids** (*list*) – List of expected IDs of the retrieved information (nodes of the index).
* **Returns:**
  Dictionary containing the following metrics:
  > - mrr (float): Mean Reciprocal Rank (MRR) metric.
  > - hit_rate (float): Hit Rate metric.
  > - precision (float): Precision metric.
  > - recall (float): Recall metric.
* **Return type:**
  dict

### *class* saga_llm_evaluation.helpers.retrieval_metrics.Relevance(llm=None)

Bases: `object`

Relevance computes the relevance score of retrieved information relatively to the query.
In other words, it validates that the retrieved information is related to the query topic.

* **Parameters:**
  **llm** (*LangChain BaseLanguageModel*) – model used for evaluation. If None,                the model is chosen as “gpt-4” by default.

#### compute(contexts: list, query: str)

This method computes the relevance score of retrieved information relatively to the query.
In other words, it validates that the retrieved information is related to the query topic.

* **Parameters:**
  * **contexts** (*list*) – List of retrieved information.
  * **query** (*str*) – Query topic.
* **Returns:**
  Relevance score for the candidate sentence. The dictionary contains the following keys:
  > - score (int): Relevance score. Binary integer value (0 or 1), where 1 indicates that the sentence is                    relevant and 0 indicates that the sentence is irrelevant.
  > - value (str): Relevance value. Y or N, where Y indicates that the sentence is relevant and N                    indicates that the sentence is irrelevant.
  > - reasoning (str): Reasoning for the relevance score.
* **Return type:**
  dict
