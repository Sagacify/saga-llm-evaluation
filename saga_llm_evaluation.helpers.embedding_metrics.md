# Embedding Metrics

These metrics are embedding-based that are used to evaluate the quality of the output of the LLM.
An embedding is a relatively low-dimensional space that captures the semantic meaning of the input text.
These embeddings are typically learned using unsupervised learning techniques such as Word2Vec, GloVe, or FastText.

As they embrace contextual information, embeddings are widely used in various NLP tasks such as text classification, sentiment analysis, and machine translation.
They can also be used to evaluate the quality of the output of the LLM, as they capture:

* The semantic meaning of the input text
* The relationships between words (grammar rules)

<a id="module-saga_llm_evaluation.helpers.embedding_metrics"></a>

### *class* saga_llm_evaluation.helpers.embedding_metrics.BERTScore(lang='en', model_type=None)

Bases: `object`

BERTScore computes a similarity score for each token in the candidate sentence with each
token in the reference sentence.
The final score is the average of the similarity scores of all tokens in the candidate sentence.

* **Parameters:**
  * **lang** (*str* *,* *optional*) – language to use. Defaults to “en”, If another language is used, a                 multilingual model is used.
  * **model_type** (*str* *,* *optional*) – Model to use. Defaults to None. If None, a default model is                 used depending on the language (see above).

#### compute(references, predictions, \*\*kwargs)

This function computes the BERTScore for each candidate sentence in the list of predictions.

* **Parameters:**
  * **references** (*list*) – List of reference sentences.
  * **predictions** (*list*) – List of candidate sentences.
* **Returns:**
  List of scores for each candidate sentence. Contains a list of scores for
  precisions, recalls, and F1 scores.
* **Return type:**
  list

### *class* saga_llm_evaluation.helpers.embedding_metrics.MAUVE(featurize_model_name='gpt2')

Bases: `object`

MAUVE score computes the difference between the candidate sentence distribution
and the reference sentence distribution. The bigger the MAUVE score, the better.

* **Parameters:**
  **featurize_model_name** (*str* *,* *optional*) – Model to use to featurize the sentences.            Defaults to “gpt2”. Check [https://huggingface.co/spaces/evaluate-metric/mauve](https://huggingface.co/spaces/evaluate-metric/mauve) for            more options.

#### compute(references, predictions, \*\*kwargs)

This function computes the MAUVE score for each candidate sentence in the list of predictions.

* **Parameters:**
  * **references** (*list*) – List of reference sentences.
  * **predictions** (*list*) – List of candidate sentences.
* **Returns:**
  List of MAUVE scores for each candidate sentence.
* **Return type:**
  list
