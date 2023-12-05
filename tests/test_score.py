import unittest

import pytest

from saga_llm_evaluation_ml.helpers.utils import load_json
from saga_llm_evaluation_ml.score import LLMScorer
from tests import LLAMA_MODEL

# skip it for github actions, too many resources needed. Test locally
pytest.skip(allow_module_level=True)


class TestLLMScorer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scorer = LLMScorer(model=LLAMA_MODEL, eval_model=LLAMA_MODEL)

    def test_init(self):
        bad_config = {"metrics": {"bert_score": {"lang": "en"}}}
        with self.assertRaises(AssertionError):
            LLMScorer(metrics=1)
            LLMScorer(config="hello")
            LLMScorer(metrics=["mauve"], config=bad_config)

    def test_score_bad_arguments(self):
        user_prompt = "I am a dog."
        knowledge = "You are a cat. You don't like dogs."
        prediction = "I am a cat, I don't like dogs."
        reference = "I am a cat, I don't like dogs, miau."
        config = load_json("saga_llm_evaluation_ml/scorer.json")

        with self.assertRaises(AssertionError):
            self.scorer.score(False, knowledge, prediction, reference, config)
            self.scorer.score(user_prompt, False, prediction, reference, config)
            self.scorer.score(user_prompt, knowledge, False, reference, config)
            self.scorer.score(user_prompt, knowledge, prediction, False, config)
            self.scorer.score([user_prompt], knowledge, prediction, reference, config)
            self.scorer.score([user_prompt], [knowledge], prediction, reference, config)
            self.scorer.score(
                [user_prompt], [knowledge], [prediction], reference, config
            )
            self.scorer.score(user_prompt, knowledge, prediction, reference, "2")
            self.scorer.score(
                [user_prompt], [knowledge, knowledge], [prediction], [reference], config
            )
            self.scorer.score(
                [user_prompt],
                [knowledge],
                [prediction, prediction],
                [reference],
                config,
            )

    def test_score(self):
        user_prompt = "I am a dog."
        knowledge = "Example: Eww, I hate dogs."
        prediction = "I am a cat, I don't like dogs."
        reference = "I am a cat, I don't like dogs, miau."
        config = load_json("saga_llm_evaluation_ml/scorer.json")

        scores = self.scorer.score(
            user_prompt=user_prompt,
            prediction=prediction,
            knowledge=knowledge,
            reference=reference,
            config=config,
        )
        self.assertTrue(isinstance(scores, dict))
        self.assertTrue("evaluation" in scores)
        self.assertTrue("metadata" in scores)
