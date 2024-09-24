<a id="usage-section"></a>

# Usage

## Default use of the Scorer

The Scorer is a class that allows you to run multiple metrics at once. The metrics supported are BERTScore, MAUVE, BLEURTScore, Q-Squared, SelCheck-GPT, G-Eval, and GPT-Score, but of course this list is likely to grow in the future.

```python
from saga_llm_evaluation import LLMScorer
from langchain_core.language_models.chat_models import BaseChatModel

scorer = LLMScorer(
    metrics = ["bertscore", "mauve", "bleurtscore", "q_squared", "selcheckgpt", "geval", "gptscore"],
    model = BaseChatModel, # language model that inherits from LangChain BaseChatModel which needs to be evaluated. Needed for SelCheck-GPT
    eval_model = BaseChatModel, # language model that inherits from LangChain BaseChatModel which is used to evaluate the model. Needed for SelCheck-GPT, G-Eval and GPT-Score.
    config = {
        "bert_score": {
            "lang": "en"
        },
        "mauve": {
            "featurize_model_name": "gpt2"
        },
        "bleurt": {
            "checkpoint": "BLEURT-tiny"
        },
        "q_squared": {
            "qa_model": "ktrapeznikov/albert-xlarge-v2-squad-v2",
            "qg_model": "mrm8488/t5-base-finetuned-question-generation-ap",
            "lang": "en",
            "single": False,
            "remove_personal": True
        },
        "selfcheckgpt": {
            "n_samples": 5
        },
        "geval": {
            "aspect": "FLU",
            "task": "diag"
        },
        "gptscore": {
            "aspect": "UND",
            "task": "diag"
        }
    } # config file that contains the parameters for each metric. If not provided, the default parameters will be used (the one in the example).
)
scorer.score(
    user_prompt = "This is the user prompt",
    prediction = "This is a candidate sentence",
    knowledge = "This is the knowledge given to the LLM as a context", # optional, needed for Q-Squared
    references = ["This is a reference sentence"], # optional, needed for BERTScore, MAUVE, BLEURTScore
    config = dict(), # config file that contains the parameters for each metric (see above). If not provided, the default parameters will be used (the one in the example).
)
```

## Standalone use of the metrics

### Embedding-based metrics

#### BERTScore

```python
from saga_llm_evaluation import BERTScore

bert_score = BERTScore()
scores = bert_score.compute(
    references=["This is a reference sentence"],
    predictions=["This is a candidate sentence"],
)
```

#### MAUVE

```python
from saga_llm_evaluation import MAUVE

mauve = MAUVE()
scores = mauve.compute(
    references=["This is a reference sentence"],
    predictions=["This is a candidate sentence"],
)
```

### Language-Model-based metrics

#### BLEURTScore

```python
from saga_llm_evaluation import BLEURTScore

bleurt_score = BLEURTScore()
scores = bleurt_score.compute(
    references=["This is a reference sentence"],
    predictions=["This is a candidate sentence"],
)
```

#### Q-Squared

```python
from saga_llm_evaluation import QSquared

q_squared = QSquared()
scores = q_squared.compute(
    user_prompts=["This is the user prompt"],
    predictions=["This is a candidate sentence"],
    knowledge="This is the knowledge given to the LLM as a context",
)
```

### LLM-based metrics

#### SelfCheck-GPT

```python
from saga_llm_evaluation import SelfCheckGPT
from langchain_core.language_models.chat_models import BaseChatModel

selfcheck_gpt = SelfCheckGPT(
    model = BaseChatModel, # language model that inherits from LangChain BaseChatModel which needs to be evaluated.
    eval_model = BaseChatModel, # language model that inherits from LangChain BaseChatModel which is used to evaluate the model.
)
scores = selfcheck_gpt.compute(
    user_prompts=["This is the user prompt"],
    predictions=["This is a candidate sentence"],
)
```

#### G-Eval

```python
from saga_llm_evaluation import GEval
from langchain_core.language_models.chat_models import BaseChatModel

g_eval = GEval(
    model = BaseChatModel, # language model that inherits from LangChain BaseChatModel which needs to be evaluated.
)
```

- Using pre-defined tasks and aspects:

```python
scores = g_eval.compute(
    user_prompts=["This is the user prompt"],
    predictions=["This is a candidate sentence"],
    task="diag", # task to evaluate
    aspects=["CON"], # aspects to evaluate (consistent, fluent, informative, interesting, relevant, specific, ...)
)
```

- Using custom tasks and aspects:

```python
scores = g_eval.compute(
    user_prompts=["This is the user prompt"],
    predictions=["This is a candidate sentence"],
    custom_prompt = {
        "name": "Fluency",
        "task": "diag",
        "aspect": "Fluency (1-5) - the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure. - 1: Poor. The summary is difficult to read and understand. It contains many grammatical errors, spelling mistakes, and/or punctuation errors. - 2: Fair. The summary is somewhat difficult to read and understand. It contains some grammatical errors, spelling mistakes, and/or punctuation errors. - 3: Good. The summary is easy to read and understand. It contains few grammatical errors, spelling mistakes, and/or punctuation errors. - 4: Very Good. The summary is easy to read and understand. It contains no grammatical errors, spelling mistakes, and/or punctuation errors. - 5: Excellent. The summary is easy to read and understand. It contains no grammatical errors, spelling mistakes, and/or punctuation errors",
    }, # custom prompt to use, you can create your own evaluation prompt.
)
```

#### GPT-Score

```python
from saga_llm_evaluation import GPTScore
from langchain_core.language_models.chat_models import BaseChatModel

gpt_score = GPTScore(
    model = BaseChatModel, # language model that inherits from LangChain BaseChatModel which needs to be evaluated.
)
```

- Using pre-defined tasks and aspects:

```python
scores = gpt_score.compute(
    user_prompts=["This is the user prompt"],
    predictions=["This is a candidate sentence"],
    task="diag", # task to evaluate
    aspects=["CON"], # aspects to evaluate (consistent, fluent, informative, interesting, relevant, specific, ...)
)
```

- Using custom tasks and aspects:

```python
scores = gpt_score.compute(
    user_prompts=["This is the user prompt"],
    predictions=["This is a candidate sentence"],
    custom_prompt = {
        "name": "FLU", #fluency
        "task": "diag",
        "aspect": "Answer the question based on the conversation between a human and AI.\nQuestion: Is the response of AI fluent throughout the conversation? (a) Yes. (b) No.\nConversation:\nUser: {{src}}\nAI: {{pred}}\nAnswer:",
    }, # custom prompt to use, you can create your own evaluation prompt.
)
```

#### Relevance

```python
from saga_llm_evaluation.helpers.language_metrics import Relevance

relevance = Relevance()
scores = relevance.compute(
    user_prompts=["This is the user prompt"],
    predictions=["This is a candidate sentence"],
)
```

#### Correctness

```python
from saga_llm_evaluation.helpers.language_metrics import Correctness

correctness = Correctness()
scores = correctness.compute(
    user_prompts=["This is the user prompt"],
    predictions=["This is a candidate sentence"],
    references=["This is the reference sentence"],
)
```

#### Faithfulness

```python
from saga_llm_evaluation.helpers.language_metrics import Faithfulness

faithfulness = Faithfulness()
scores = faithfulness.compute(
    user_prompts=["This is the user prompt"],
    predictions=["This is a candidate sentence"],
    references=["This is the reference sentence"],
)
```

#### Negative Rejection

```python
from saga_llm_evaluation.helpers.language_metrics import NegativeRejection

negative_rejection = NegativeRejection()
scores = negative_rejection.compute(
    user_prompts=["This is the user prompt"],
    predictions=["This is a candidate sentence"],
    references=["This is the reference sentence"],
)
```

#### HallucinationScore

```python
from saga_llm_evaluation.helpers.language_metrics import HallucinationScore

hallucination_score = HallucinationScore()
scores = hallucination_score.compute(
    predictions=["This is a candidate sentence"],
    references=["This is the reference sentence"],
)
```

### Retrieval-based metrics

#### Relevance

```python
from saga_llm_evaluation.helpers.retrieval_metrics import Relevance

relevance = Relevance()
scores = relevance.compute(
    contexts=["This is the retrieved information"],
    query="This is the query topic",
)
```

#### Accuracy

```python
from saga_llm_evaluation.helpers.retrieval_metrics import Accuracy
from llama_index.core import VectorStoreIndex

# Assuming you have an index created and populated with the relevant data
index = VectorStoreIndex()

accuracy = Accuracy(index=index, k=2)
scores = accuracy.compute(
    query="This is the query topic",
    expected_ids=["expected_id_1", "expected_id_2"],
)
```

### Using a different LangChain model as evaluator

You can use a different model as evaluator by using any model that inherits from LangChain [BaseLanguageModel](https://api.python.langchain.com/en/latest/language_models/langchain_core.language_models.chat_models.BaseChatModel.html). This is the preffered way to use the metrics. LangChain offers a wide range of models that can be used as evaluator. However, if a model you want to use is not available, you can still define your own evaluator model, see this [tutorial](https://python.langchain.com/docs/how_to/custom_chat_model/).
