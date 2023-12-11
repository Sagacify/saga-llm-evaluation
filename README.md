# 🔮 Sagacify LLM Evaluation ML Library 🔮

Welcome to the Saga LLM Evaluation ML library, a versatile Python library designed for evaluating the performance of large language models in Natural Language Processing (NLP) tasks. Whether you're developing language models, chatbots, or other NLP applications, our library provides a comprehensive suite of metrics to help you assess the quality of your language models.

It is build on top of the [Hugging Face Transformers](https://github.com/huggingface/transformers) library with some additional metrics and features. We divided the metrics into three categories: embedding-based, language-model-based, and LLM-based metrics. You can use the metrics individually or all at once using the Scorer provided by this library, depending on the availability of references, context, and other parameters.

Moreover, The Scorer function provides metafeatures that are extracted from the prompt, prediction, and knowledge via the [Elementa Library](https://docs.elemeta.ai/metafeatures.html). This allows you to monitor the performance of your model based on the structure of the prompt, prediction, and knowledge.

Developed by [Sagacify](https://sagacify.com/)

## Available Metrics
### Embedding-based Metrics
- **BERTScore**: A metric that measures the similarity between model-generated text and human-generated references. It leverages the pre-trained contextual embeddings from BERT and matches words in candidate and reference sentences by cosine similarity It's a valuable tool for evaluating semantic content. [Read more](https://arxiv.org/pdf/1904.09675.pdf).
- **MAUVE**: Compute the divergence between the learned distributions from a text generated from a text generation model and human written text. [Read more](https://arxiv.org/pdf/2212.14578.pdf).

### Language-Model-based Metrics
- **BLEURTScore**: It is a learned evaluation metric for Natural Language Generation. It is built using multiple phases of transfer learning starting from a pretrained BERT model, employing another pre-training phrase using synthetic data, and finally trained on WMT human annotations. [Read more](https://arxiv.org/pdf/2004.04696.pdf).
- **Q-Squared**: Reference-free metric that aims to evaluate the factual consistency of knowledge-grounded dialogue systems.The approach is based on automatic question generation and question answering. More specifically, it generates questions from the knowledge base and uses the generated questions to evaluate the factual consistency of the generated response. [Read more](https://arxiv.org/pdf/2104.08202.pdf).

### LLM-based Metrics
- **SelCheck-GPT** (QA approach): A metric that evaluates the correctness of language model outputs by identifying comparing the output to the typical distribution of the model outputs. It introduces a zero-fashion approach to fact-check the response of black box models and assess hallucination problems. [Read more](https://arxiv.org/pdf/2303.08896.pdf).
- **G-Eval**: A framework that use LLMs with chain-of-thoughts (CoT) and a form-filling paradigm, to assess the quality of NLG outputs. It has been experimented with two generation tasks, text summarization and dialogue generation and many evaluation criteria. The task and the evaluation criteria may be changed depending on the application. [Read more](arxiv.org/pdf/2303.16634.pdf).
- **GPT-Score**: Evaluation framework which utilizes the emergent abilities (e.g., zero-shot instruction) of generative pre-trained models to score generated texts. Experimental results on four text generation tasks, 22 evaluation aspects, and corresponding 37 datasets demonstrate that this approach can effectively allow us to achieve what one desires to evaluate for texts simply by natural language instructions [Read more](https://arxiv.org/pdf/2302.04166.pdf).

Each of these metrics uses the [LLAMA model](https://ai.meta.com/llama/) to evaluate the performance of the LLM that needs to be evaluated. Particularly, we use a quantized version of the LLAMA model to reduce the inference time, available [here](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF). As we use the most lightweight version of the LLAMA model by default, you can expect a better performance if you use other versions of the LLAMA model. The ones we tested for the experiments are available [here](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF)


## Installation
To install the Saga LLM Evaluation ML library, use the following command:

```pip install sagacify-llm-evaluation```

Be aware that by default the library will run pytorch on the CPU. If you want to run it on the GPU, you need to install pytorch with GPU support. You can find the instructions [here](https://pytorch.org/get-started/locally/).

## Usage

### Default use of the Scorer

The Scorer is a class that allows you to run multiple metrics at once. The metrics supported are BERTScore, MAUVE, BLEURTScore, Q-Squared, SelCheck-GPT, G-Eval, and GPT-Score.


```python
from sagacify_llm_evaluation import LLMScorer
scorer = LLMScorer(
    metrics = ["bertscore", "mauve", "bleurtscore", "q_squared", "selcheckgpt", "geval", "gptscore"],
    model = transformers.PreTrainedModel, # language model that inherits from transformers.PreTrainedModel which needs to be evaluated. Needed for SelCheck-GPT
    eval_model = transformers.PreTrainedModel, # language model that inherits from transformers.PreTrainedModel which is used to evaluate the model. Needed for SelCheck-GPT, G-Eval and GPT-Score.
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
            "model_name_or_path": "TheBloke/Llama-2-7b-Chat-GGUF", 
            "model_basename": "llama-2-7b-chat.Q2_K.gguf",
            "n_samples": 5
        },
        "geval": {
            "model_name_or_path": "TheBloke/Llama-2-7b-Chat-GGUF", 
            "model_basename": "llama-2-7b-chat.Q2_K.gguf",
            "aspect": "FLU",
            "task": "diag"
        },
        "gptscore": {
            "eval_model_name_or_path": "TheBloke/Llama-2-7b-Chat-GGUF", 
            "eval_model_basename": "llama-2-7b-chat.Q2_K.gguf",
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

### BERTScore
```python
from sagacify_llm_evaluation import BERTScore

bert_score = BERTScore()
scores = bert_score.compute(
    references=["This is a reference sentence"],
    predictions=["This is a candidate sentence"],
)
```

### MAUVE
```python
from sagacify_llm_evaluation import MAUVE
mauve = MAUVE()
scores = mauve.compute(
    references=["This is a reference sentence"],
    predictions=["This is a candidate sentence"],
)
```

### BLEURTScore
```python
from sagacify_llm_evaluation import BLEURTScore
bleurt_score = BLEURTScore()
scores = bleurt_score.compute(
    references=["This is a reference sentence"],
    predictions=["This is a candidate sentence"],
)
```

### Q-Squared
```python
from sagacify_llm_evaluation import QSquared
q_squared = QSquared()
scores = q_squared.compute(
    knowledges=["This is the text gave to the LLM as knowledge"],
    predictions=["This is a candidate sentence"],
)
```

### SelCheck-GPT
```python
from sagacify_llm_evaluation import SelCheckGPT
selcheck_gpt = SelCheckGPT(
    model = transformers.PreTrainedModel, # language model that inherits from transformers.PreTrainedModel which needs to be evaluated.
    eval_model = transformers.PreTrainedModel, # language model that inherits from transformers.PreTrainedModel which is used to evaluate the model.
)
scores = selcheck_gpt.compute(
    user_prompts=["This is the user prompt"],
    predictions=["This is a candidate sentence"],
)
```

### G-Eval
```python
from sagacify_llm_evaluation import GEval
g_eval = GEval(
    model = transformers.PreTrainedModel, # language model that inherits from transformers.PreTrainedModel which is used to evaluate the model.
)

# Using pre-defined tasks and aspects
scores = g_eval.compute(
    user_prompts=["This is the user prompt"],
    predictions=["This is a candidate sentence"],
    task="diag", # task to evaluate
    aspects=["CON"], # aspects to evaluate (consistent, fluent, informative, interesting, relevant, specific, ...)
)

# Using custom tasks and aspects
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

### GPT-Score
```python
from sagacify_llm_evaluation import GPTScore
gpt_score = GPTScore(
    model = transformers.PreTrainedModel, # language model that inherits from transformers.PreTrainedModel which is used to evaluate the model.
)

# Using pre-defined tasks and aspects
scores = gpt_score.compute(
    user_prompts=["This is the user prompt"],
    predictions=["This is a candidate sentence"],
    task="diag", # task to evaluate
    aspects=["CON"], # aspects to evaluate (consistent, fluent, informative, interesting, relevant, specific, ...)
)

# Using custom tasks and aspects
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

### Using a different LLAMA model as evaluator
You can use a different LLAMA model as evaluator by using the get_llama_model function. This function will download the LLAMA model and return a transformers.PreTrainedModel that you can use as evaluator in the SelCheck-GPT, G-Eval, and GPT-Score metrics.

The full list of quantized LLAMA models that may be used is available [here](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF).

```python
from sagacify_llm_evaluation import get_llama_model

llama_model = get_llama_model(
    repo_id = "TheBloke/Llama-2-7b-Chat-GGUF", 
    filename = "llama-2-7b-chat.Q2_K.gguf",
)
```

You can also download the LLAMA model manually and specify the local path to the model in the get_llama_model function.

```huggingface-cli download TheBloke/Llama-2-7b-Chat-GGUF llama-2-7b-chat.Q2_K.gguf --local-dir path_to_model_folder --local-dir-use-symlinks False```

```python
from sagacify_llm_evaluation import get_llama_model

llama_model = get_llama_model(
    model_path = "path_to_model_folder", 
)
```


Feel free to contribute and make this library even more powerful! We appreciate your support. 💻💪🏻

API documentation coming soon!