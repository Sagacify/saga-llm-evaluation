import unittest

from unittest.mock import MagicMock, patch
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.base import BaseLanguageModel

from saga_llm_evaluation.helpers.llm_metrics import (
    GEval,
    GPTScore,
    SelfCheckGPT,
    Relevance,
    Correctness,
    Faithfulness,
    NegativeRejection,
)


class TestGEvalWithMockedModel(unittest.TestCase):
    def setUp(self):
        patcher = patch("saga_llm_evaluation.helpers.utils.get_langchain_gpt_model")
        mock_get_model = patcher.start()

        self.mock_model = MagicMock(spec=BaseChatModel)
        mock_get_model.return_value = self.mock_model

        mock_response = MagicMock()
        mock_response.response_metadata = {
            "logprobs": {
                "content": [
                    {"token": "1", "logprob": -0.5},
                    {"token": ".", "logprob": -0.1},
                    {"token": "good", "logprob": -0.3},
                ]
            }
        }
        self.mock_model.invoke.return_value = mock_response
        self.geval = GEval(model=self.mock_model)

        self.addCleanup(patcher.stop)

    def test_compute_with_mock_model(self):
        source = "Hello, how are you?"
        pred = "I'm good, thanks!"
        task = "diag"
        aspect = "ENG"

        result = self.geval.compute(source, pred, task, aspect)
        self.assertTrue(result["ENG"][0] > 0)
        self.assertAlmostEqual(result["ENG"][0], 0.6065306597126334)


class TestSelfCheckGPTWithMockedModel(unittest.TestCase):
    def setUp(self):
        patcher = patch("saga_llm_evaluation.helpers.utils.get_langchain_gpt_model")
        mock_get_model = patcher.start()

        self.mock_model = MagicMock(spec=BaseChatModel)
        mock_get_model.return_value = self.mock_model

        mock_response = MagicMock()
        mock_response.content = "Yes"
        self.mock_model.invoke.return_value = mock_response

        self.selfcheckgpt = SelfCheckGPT(
            model=self.mock_model, eval_model=self.mock_model
        )

        self.addCleanup(patcher.stop)

    def test_compute_with_mock_model(self):
        user_prompt = "What is the capital of France?"
        prediction = "Paris"
        n_samples = 3

        result = self.selfcheckgpt.compute(user_prompt, prediction, n_samples)
        self.assertEqual(result[0], 1)


class TestGPTScoreWithMockedModel(unittest.TestCase):
    def setUp(self):
        patcher = patch("saga_llm_evaluation.helpers.utils.get_langchain_gpt_model")
        mock_get_model = patcher.start()

        self.mock_model = MagicMock(spec=BaseChatModel)
        mock_get_model.return_value = self.mock_model
        self.mock_model.bind.return_value = self.mock_model

        mock_response = MagicMock()
        mock_response.response_metadata = {
            "logprobs": {
                "content": [
                    {"token": "I", "logprob": -0.5},
                    {"token": "love", "logprob": -0.1},
                    {"token": "cats", "logprob": -0.3},
                    {"token": ".", "logprob": -0.1},
                ]
            }
        }
        self.mock_model.invoke.return_value = mock_response
        self.gptscore = GPTScore(model=self.mock_model)

        self.addCleanup(patcher.stop)

    def test_compute_with_mock_model(self):
        source = "How are you?"
        pred = "I'm doing great!"
        aspect = "LIK"
        task = "diag"

        result = self.gptscore.compute(source, pred, aspect=aspect, task=task)
        self.assertAlmostEqual(result["LIK"][0], -0.25)


class TestRelevanceWithMockedModel(unittest.TestCase):
    def setUp(self):
        # Patch load_evaluator
        patcher = patch("langchain.evaluation.load_evaluator")
        mock_load_evaluator = patcher.start()

        # Patch the ChatOpenAI model
        self.llm_patcher = patch(
            "langchain_core.language_models.base.BaseLanguageModel"
        )
        self.mock_llm = self.llm_patcher.start()

        # Mock the LLM (ChatOpenAI) so it doesn't try to connect to OpenAI
        self.mock_llm_model = MagicMock(spec=BaseLanguageModel)
        self.mock_llm.return_value = self.mock_llm_model

        self.mock_evaluator = MagicMock()
        mock_load_evaluator.return_value = self.mock_evaluator

        self.mock_evaluator.evaluate_strings.return_value = {
            "score": 1,
            "value": "Y",
            "reasoning": "OK.",
        }

        self.relevance = Relevance(llm=self.mock_llm_model)
        self.relevance.evaluator = self.mock_evaluator

        self.addCleanup(patcher.stop)
        self.addCleanup(self.llm_patcher.stop)

    def test_compute_with_mock_model(self):
        user_prompts = ["What is the capital of France?"]
        predictions = ["Paris is the capital of France."]

        result = self.relevance.compute(user_prompts, predictions)
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["value"], "Y")
        self.assertEqual(result["reasoning"], "OK.")


class TestCorrectnessWithMockedModel(unittest.TestCase):
    def setUp(self):
        patcher = patch("langchain.evaluation.load_evaluator")
        mock_load_evaluator = patcher.start()

        # Patch the llm model
        self.llm_patcher = patch(
            "langchain_core.language_models.base.BaseLanguageModel"
        )
        self.mock_llm = self.llm_patcher.start()

        # Mock the LLM (ChatOpenAI) so it doesn't try to connect to OpenAI
        self.mock_llm_model = MagicMock(spec=BaseLanguageModel)
        self.mock_llm.return_value = self.mock_llm_model

        # Mock the evaluator
        self.mock_evaluator = MagicMock()
        mock_load_evaluator.return_value = self.mock_evaluator

        self.mock_evaluator.evaluate_strings.return_value = {
            "score": 1,
            "value": "Y",
            "reasoning": "OK.",
        }

        self.correctness = Correctness(llm=self.mock_llm_model)
        self.correctness.evaluator = self.mock_evaluator

        self.addCleanup(patcher.stop)
        self.addCleanup(self.llm_patcher.stop)

    def test_compute_with_mock_model(self):
        user_prompts = ["What is the capital of France?"]
        predictions = ["Paris is the capital of France."]
        references = ["Paris is the capital of France."]

        result = self.correctness.compute(user_prompts, predictions, references)
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["value"], "Y")
        self.assertEqual(result["reasoning"], "OK.")


class TestFaithfulnessWithMockedModel(unittest.TestCase):
    def setUp(self):

        # Patch load_evaluator
        patcher = patch("langchain.evaluation.load_evaluator")
        mock_load_evaluator = patcher.start()

        # Patch the llm model
        self.llm_patcher = patch(
            "langchain_core.language_models.base.BaseLanguageModel"
        )
        self.mock_llm = self.llm_patcher.start()

        # Mock the LLM (ChatOpenAI) so it doesn't try to connect to OpenAI
        self.mock_llm_model = MagicMock(spec=BaseLanguageModel)
        self.mock_llm.return_value = self.mock_llm_model

        # Mock the evaluator
        self.mock_evaluator = MagicMock()
        mock_load_evaluator.return_value = self.mock_evaluator
        self.mock_evaluator.evaluate_strings.return_value = {
            "score": 1,
            "value": "Y",
            "reasoning": "OK.",
        }

        self.faithfulness = Faithfulness(llm=self.mock_llm_model)
        self.faithfulness.evaluator = self.mock_evaluator

        self.addCleanup(patcher.stop)
        self.addCleanup(self.llm_patcher.stop)

    def test_compute_with_mock_model(self):
        user_prompts = ["What is the capital of France?"]
        predictions = ["Paris is the capital of France."]
        references = ["Paris is the capital of France."]

        result = self.faithfulness.compute(user_prompts, predictions, references)
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["value"], "Y")
        self.assertEqual(result["reasoning"], "OK.")


class TestNegativeRejectionWithMockedModel(unittest.TestCase):
    def setUp(self):
        # Patch load_evaluator
        patcher = patch("langchain.evaluation.load_evaluator")
        mock_load_evaluator = patcher.start()

        # Patch the llm model
        self.llm_patcher = patch(
            "langchain_core.language_models.base.BaseLanguageModel"
        )
        self.mock_llm = self.llm_patcher.start()

        # Mock the LLM (ChatOpenAI) so it doesn't try to connect to OpenAI
        self.mock_llm_model = MagicMock(spec=BaseLanguageModel)
        self.mock_llm.return_value = self.mock_llm_model

        # Mock the evaluator
        self.mock_evaluator = MagicMock()
        mock_load_evaluator.return_value = self.mock_evaluator
        self.mock_evaluator.evaluate_strings.return_value = {
            "score": 1,
            "value": "Y",
            "reasoning": "OK.",
        }

        self.negative_rejection = NegativeRejection(llm=self.mock_llm_model)
        self.negative_rejection.evaluator = self.mock_evaluator

        self.addCleanup(patcher.stop)
        self.addCleanup(self.llm_patcher.stop)

    def test_compute_with_mock_model(self):
        user_prompts = ["Tell me about unicorns."]
        predictions = ["I cannot provide information on that topic."]
        references = ["There is no information on unicorns."]

        result = self.negative_rejection.compute(user_prompts, predictions, references)
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["value"], "Y")
        self.assertEqual(result["reasoning"], "OK.")
