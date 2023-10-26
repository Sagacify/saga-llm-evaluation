# 🔮 Saga LLM Evaluation ML Library

[![ci](https://github.com/sagacify/saga-llm-evaluation-ml/actions/workflows/cd.yaml/badge.svg)](https://github.com/Sagacify/saga-llm-evaluation-ml/actions/workflows/cd.yaml)

Saga LLM Evaluation ML is a Python library to evaluate the performance of any kind of large language models for NLP tasks.
Some of the metrics include:
- Embedding-based metrics
  - [BERTScore](https://arxiv.org/pdf/1904.09675.pdf)
  - [MAUVE](https://arxiv.org/pdf/2212.14578.pdf)
- Language-based metrics
  - [BLEURTScore](https://arxiv.org/pdf/2004.04696.pdf)
  - Q-Squared
- LLM-based metrics
  - [SelCheck-GPT](https://arxiv.org/pdf/2303.08896.pdf)
  - [G-Eval](https://arxiv.org/pdf/2303.16634.pdf)
  - [GPT-Score](https://arxiv.org/pdf/2302.04166.pdf)

These metrics can be assessed all at once using the Scorer implemented by this library, but also individually. The metrics computed by the scorer depend on the availability of a reference, context, and other parameters.

## Installation

```poetry add saga_llm_evaluation_ml```

## Usage
Complete example of usage:

### Default use of the Scorer
```python
from saga_llm_evaluation_ml.score import Scorer
scorer = Scorer()
results = scorer.score()
```

### Standalone use of the metrics
```python
from saga_llm_evaluation_ml.helpers.embedding_metrics import BERTScore

bert_score = BERTScore()
scores = bert_score.compute_score(
    references=["This is a reference sentence"],
    candidates=["This is a candidate sentence"],
)
```

## Development

To setup the project, clone and execute:

`poetry install`

You can create a test script `test.py` at the root of the directory and use this to test:

`poetry run python test.py`

### Run tests

To run the tests:

```sh
poetry run pytest
```

### Run linter and formatter

```
# Formatter
poetry run black saga_llm_evaluation_ml/ tests/
# Linter
poetry run pylint saga_llm_evaluation_ml/ tests/
```