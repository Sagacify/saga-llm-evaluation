# API Reference

## Helpers Module

* [Metrics](saga_llm_evaluation.helpers.md)
  * [Embedding Metrics](saga_llm_evaluation.helpers.embedding_metrics.md)
  * [LLM Metrics](saga_llm_evaluation.helpers.llm_metrics.md)
  * [Language Metrics](saga_llm_evaluation.helpers.language_metrics.md)
  * [Retrieval Metrics](saga_llm_evaluation.helpers.retrieval_metrics.md)
* [Utils](saga_llm_evaluation.helpers.md#module-saga_llm_evaluation.helpers.utils)
  * [`MetadataExtractor`](saga_llm_evaluation.helpers.md#saga_llm_evaluation.helpers.utils.MetadataExtractor)
  * [`check_list_type()`](saga_llm_evaluation.helpers.md#saga_llm_evaluation.helpers.utils.check_list_type)
  * [`clean_text()`](saga_llm_evaluation.helpers.md#saga_llm_evaluation.helpers.utils.clean_text)
  * [`filter_class_input()`](saga_llm_evaluation.helpers.md#saga_llm_evaluation.helpers.utils.filter_class_input)
  * [`filter_questions()`](saga_llm_evaluation.helpers.md#saga_llm_evaluation.helpers.utils.filter_questions)
  * [`get_langchain_gpt_model()`](saga_llm_evaluation.helpers.md#saga_llm_evaluation.helpers.utils.get_langchain_gpt_model)
  * [`get_langchain_llama_model()`](saga_llm_evaluation.helpers.md#saga_llm_evaluation.helpers.utils.get_langchain_llama_model)
  * [`get_llama_model()`](saga_llm_evaluation.helpers.md#saga_llm_evaluation.helpers.utils.get_llama_model)
  * [`load_json()`](saga_llm_evaluation.helpers.md#saga_llm_evaluation.helpers.utils.load_json)
  * [`non_personal()`](saga_llm_evaluation.helpers.md#saga_llm_evaluation.helpers.utils.non_personal)
  * [`raw_f1_score()`](saga_llm_evaluation.helpers.md#saga_llm_evaluation.helpers.utils.raw_f1_score)

## Scorer Module

### *class* saga_llm_evaluation.score.LLMScorer(metrics=['bert_score', 'mauve', 'bleurt', 'q_squared', 'selfcheckgpt', 'geval', 'gptscore'], model=None, eval_model=None, config=None)

Bases: `object`

Initialize the LLMScorer class. This class is used to evaluate the performance of a language model        using a set of evaluation metrics.        The evaluation metrics are defined in the config file or can be passed as an input parameter.        The model to evaluate and the evaluation model can be passed as input parameters.

* **Parameters:**
  * **metrics** (*list* *,* *optional*) – List of evaluation metrics to use.                Defaults to [“bert_score”, “mauve”, “bleurt”, “q_squared”, “selfcheckgpt”, “geval”, “gptscore”].
  * **model** (*object* *,* *optional*) – Model to evaluate. Defaults to None.
  * **eval_model** (*object* *,* *optional*) – Evaluation model. Defaults to None.
  * **config** (*dict* *,* *optional*) – Config file. Defaults to None.

#### add_geval_aspect(code: str, name: str, prompt: str)

This function adds a new aspect to the GEval metric.
Please follow the example pattern below to ensure consistency.

Example:

```python
"COH": {
    "name": "Coherence",
    "prompt": "Coherence (1-5) - the overall quality and logical flow of all sentences.\
        This dimension aligns with the DUC quality question of structure and coherence, which states that\
        the summary should be well-structured and well-organized. It should not just be\
        a collection of related information, but should build from sentence to sentence\
        to form a coherent body of information about a topic."
}
```

* **Parameters:**
  * **code** (*str*) – Code of the aspect.
  * **name** (*str*) – Name of the aspect.
  * **prompt** (*str*) – Prompt of the aspect.

#### add_geval_task(name: str, definition: str)

This function adds a new task to the GEval metric.
Please follow the example pattern below to ensure consistency.

Example:

```python
"summ": "You will be given one summary written for a news article.\n"
        "Your task is to rate the summary on one metric.\n"
        "Please make sure you read and understand these instructions carefully.\n"
        "Please keep this document open while reviewing, and refer to it as needed."
```

* **Parameters:**
  * **name** (*str*) – Name of the task.
  * **definition** (*str*) – Definition of the task.

#### add_gptscore_template(task: str, code: str, prompt: str)

This function adds a template to the GPTScore metric.
Please follow the example pattern below to ensure consistency.

Example:

```python
"diag": {
    "COH": (
        f"Answer the question based on the conversation between a human and AI.\n"
        "Question: Is the AI coherent and maintains a good conversation flow throughout the conversation? (a) Yes. (b) No.\n"
        "Conversation:\nUser: {{src}}\nAI: {{pred}}\nAnswer: Yes."
    ),
}
```

* **Parameters:**
  * **task** (*str*) – Task of the template.
  * **code** (*str*) – Code of the aspect.
  * **prompt** (*str*) – Prompt of the aspect.

#### score(user_prompt: list, prediction: list, knowledge: list | None = None, reference: list | None = None, config: dict | None = None)

This function computes the evaluation metrics for a given user prompt and prediction.

* **Parameters:**
  * **user_prompt** (*str*) – user prompt to the model
  * **prediction** (*str*) – Prediction of the model.
  * **knowledge** (*str* *,* *optional*) – Source text that the model used to generate the prediction. Defaults to None.
  * **reference** (*str* *,* *optional*) – Reference of the prediction. Defaults to None.
  * **config** (*dict* *,* *optional*) – Config file. Defaults to None.
* **Returns:**
  Dictionary containing the metadata and evaluation metrics.
* **Return type:**
  dict

### saga_llm_evaluation.score.get_model(config: dict, model=None, eval_model=None, key: str = 'bert_score')

Get the evaluation metric model.

* **Parameters:**
  * **config** (*dict*) – Config file.
  * **model** (*object* *,* *optional*) – Model to evaluate. Defaults to None.
  * **eval_model** (*object* *,* *optional*) – Evaluation model. Defaults to None.
  * **key** (*str* *,* *optional*) – Evaluation metric to use. Defaults to “bert_score”.
* **Returns:**
  Evaluation metric model.
* **Return type:**
  object
