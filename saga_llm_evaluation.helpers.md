# Metrics

* [Embedding Metrics](saga_llm_evaluation.helpers.embedding_metrics.md)
  * [`BERTScore`](saga_llm_evaluation.helpers.embedding_metrics.md#saga_llm_evaluation.helpers.embedding_metrics.BERTScore)
  * [`MAUVE`](saga_llm_evaluation.helpers.embedding_metrics.md#saga_llm_evaluation.helpers.embedding_metrics.MAUVE)
* [LLM Metrics](saga_llm_evaluation.helpers.llm_metrics.md)
  * [`Correctness`](saga_llm_evaluation.helpers.llm_metrics.md#saga_llm_evaluation.helpers.llm_metrics.Correctness)
  * [`Faithfulness`](saga_llm_evaluation.helpers.llm_metrics.md#saga_llm_evaluation.helpers.llm_metrics.Faithfulness)
  * [`GEval`](saga_llm_evaluation.helpers.llm_metrics.md#saga_llm_evaluation.helpers.llm_metrics.GEval)
  * [`GPTScore`](saga_llm_evaluation.helpers.llm_metrics.md#saga_llm_evaluation.helpers.llm_metrics.GPTScore)
  * [`HallucinationScore`](saga_llm_evaluation.helpers.llm_metrics.md#saga_llm_evaluation.helpers.llm_metrics.HallucinationScore)
  * [`NegativeRejection`](saga_llm_evaluation.helpers.llm_metrics.md#saga_llm_evaluation.helpers.llm_metrics.NegativeRejection)
  * [`Relevance`](saga_llm_evaluation.helpers.llm_metrics.md#saga_llm_evaluation.helpers.llm_metrics.Relevance)
  * [`SelfCheckGPT`](saga_llm_evaluation.helpers.llm_metrics.md#saga_llm_evaluation.helpers.llm_metrics.SelfCheckGPT)
* [Language Metrics](saga_llm_evaluation.helpers.language_metrics.md)
  * [`BLEURTScore`](saga_llm_evaluation.helpers.language_metrics.md#saga_llm_evaluation.helpers.language_metrics.BLEURTScore)
  * [`QSquared`](saga_llm_evaluation.helpers.language_metrics.md#saga_llm_evaluation.helpers.language_metrics.QSquared)
* [Retrieval Metrics](saga_llm_evaluation.helpers.retrieval_metrics.md)
  * [`Accuracy`](saga_llm_evaluation.helpers.retrieval_metrics.md#saga_llm_evaluation.helpers.retrieval_metrics.Accuracy)
  * [`Relevance`](saga_llm_evaluation.helpers.retrieval_metrics.md#saga_llm_evaluation.helpers.retrieval_metrics.Relevance)

# Utils

### *class* saga_llm_evaluation.helpers.utils.MetadataExtractor

Bases: `object`

Initializes the metadata extractor.

#### add_custom_extractor(extractor: AbstractMetafeatureExtractor)

Adds a custom extractor to the metadata extractor.

* **Parameters:**
  **extractor** (*object*) – extractor to add

#### add_regex_match_count(regex_rule, name=None)

Adds a regex rule to the metadata extractor.
For a given regex return the number of matches it has in the text.

* **Parameters:**
  **regex_rule** (*str*) – regex rule to add

#### add_word_regex_matches_count(regex_rule, name=None)

Adds a regex rule to the metadata extractor.
For a given regex return the number of words matching the regex.

* **Parameters:**
  **regex_rule** (*str*) – regex rule to add

#### compute(text)

Computes metadata from a text using elemeta library and returns a dictionary of metadata.

* **Parameters:**
  **text** (*str*) – text to extract metadata from
* **Returns:**
  dictionary of metadata
* **Return type:**
  dict

### saga_llm_evaluation.helpers.utils.check_list_type(array: list, list_type: type)

Check if an array is a list of a given type.

* **Parameters:**
  * **array** (*list*) – array to check
  * **list_type** (*type*) – type to check
* **Returns:**
  True if the array is a list of the given type, False otherwise
* **Return type:**
  bool

### saga_llm_evaluation.helpers.utils.clean_text(text)

Clean a text by removing punctuation and (some) stopwords.

* **Parameters:**
  **text** (*str*) – text to clean
* **Returns:**
  cleaned text
* **Return type:**
  str

### saga_llm_evaluation.helpers.utils.filter_class_input(args, python_function: object, drop=None)

Filters input arguments for a given class.

* **Parameters:**
  * **args** (*dict*) – dictionary of arguments
  * **python_class** (*object*) – class to filter arguments for
  * **drop** (*list* *,* *optional*) – list of arguments to drop. Defaults to None.
* **Returns:**
  filtered dictionary of arguments
* **Return type:**
  dict

### saga_llm_evaluation.helpers.utils.filter_questions(exp_ans, pred_ans)

Check if the expected answer and the predicted answer are the same.

* **Parameters:**
  * **exp_ans** (*str*) – expected answer
  * **pred_ans** (*str*) – predicted answer
* **Returns:**
  “VALID” if the answers are the same, “NO MATCH” otherwise
* **Return type:**
  str

### saga_llm_evaluation.helpers.utils.get_langchain_gpt_model(version='gpt-3.5-turbo-0125')

Return a GPT model from Langchain OpenAI.

* **Parameters:**
  **version** (*str*) – model version
* **Returns:**
  GPT model from LangChain OpenAI.
* **Return type:**
  langchain_openai.ChatOpenAI

### saga_llm_evaluation.helpers.utils.get_langchain_llama_model(repo_id: str = 'TheBloke/Llama-2-7b-Chat-GGUF', filename: str = 'llama-2-7b-chat.Q2_K.gguf', model_path=False)

Download and return a LlamaCPP model from LangChain, loaded from the HuggingFace Hub.

* **Parameters:**
  * **repo_id** (*str*) – HuggingFace Hub repo id of the model. Defaults to “TheBloke/Llama-2-7b-Chat-GGUF”.
  * **filename** (*str*) – model filename to download. Defaults to “llama-2-7b-chat.Q2_K.gguf”.
  * **model_path** (*str*) – path to the model locally to avoid downloading. Defaults to False.
* **Returns:**
  LlamaCPP model from LangChain.
* **Return type:**
  langchain_community.chat_models.ChatLlamaCpp

### saga_llm_evaluation.helpers.utils.get_llama_model(repo_id: str = 'TheBloke/Llama-2-7b-Chat-GGUF', filename: str = 'llama-2-7b-chat.Q2_K.gguf', model_path=False)

Download and return a Llama model from HuggingFace Hub.

* **Parameters:**
  * **repo_id** (*str*) – HuggingFace Hub repo id of the model. Defaults to “TheBloke/Llama-2-7b-Chat-GGUF”.
  * **filename** (*str*) – model filename to download. Defaults to “llama-2-7b-chat.Q2_K.gguf”.
  * **model_path** (*str*) – path to the model locally to avoid downloading. Defaults to False.
* **Returns:**
  Llama model
* **Return type:**
  llama_cpp.Llama

### saga_llm_evaluation.helpers.utils.load_json(path)

Load a json file from a given path.

* **Parameters:**
  **path** (*str*) – path to the json file
* **Returns:**
  dictionary of the json file
* **Return type:**
  dict

### saga_llm_evaluation.helpers.utils.non_personal(question, nlp, lan='en')

Check if a question contains personal pronouns.

* **Parameters:**
  * **question** (*str*) – question to check
  * **nlp** (*spacy.lang*) – spacy language model
  * **lan** (*str*) – language of the question. Defaults to “en”.
* **Returns:**
  True if the question does not contain personal pronouns, False otherwise
* **Return type:**
  bool

### saga_llm_evaluation.helpers.utils.raw_f1_score(a_gold, a_pred)

Compute the raw F1 score between two answers.

* **Parameters:**
  * **a_gold** (*str*) – expected answer
  * **a_pred** (*str*) – predicted answer
* **Returns:**
  F1 score
* **Return type:**
  float
