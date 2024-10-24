# Installation

To install the Saga LLM Evaluation ML library, use the following command:

```bash
pip install saga-llm-evaluation
```

If you want to use the Scorer class, you need to install the following dependencies:

```bash
pip install "saga-llm-evaluation[scorer]"
```

However, beware that this will downgrade the version of the `pandas` library to 1.5.3 if you have a higher version installed. This is due to a compatibility issue with the `pandas` library.

Be aware that by default the library will run pytorch on the CPU. If you want to run it on the GPU, you need to install pytorch with GPU support. You can find the instructions [here](https://pytorch.org/get-started/locally/).

Moreover, to use BLEURTScore, you first need to install BLEURT from the official github repository. You can install it using the following command:

```bash
pip install git+https://github.com/google-research/bleurt.git
```

Finally, spaCy is required for some metrics. You can install spaCy language models as follows (as of now, only English and French are supported):

```bash
python -m spacy download en_core_web_sm fr-core-news-sm
```
