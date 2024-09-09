import unittest
from unittest.mock import patch, MagicMock

from saga_llm_evaluation.helpers.retrieval_metrics import Relevance, Accuracy

from langchain_core.language_models.base import BaseLanguageModel


class TestRelevanceWithMockedEvaluator(unittest.TestCase):
    def setUp(self):
        # Patch load_evaluator
        self.load_evaluator_patcher = patch("langchain.evaluation.load_evaluator")
        self.mock_load_evaluator = self.load_evaluator_patcher.start()

        # Patch the ChatOpenAI model
        self.llm_patcher = patch(
            "langchain_core.language_models.base.BaseLanguageModel"
        )
        self.mock_llm = self.llm_patcher.start()

        # Mock the LLM (ChatOpenAI) so it doesn't try to connect to OpenAI
        self.mock_llm_model = MagicMock(spec=BaseLanguageModel)
        self.mock_llm.return_value = self.mock_llm_model

        # Mock the evaluator
        self.mock_evaluator = MagicMock()  # Use the correct evaluator spec
        self.mock_load_evaluator.return_value = self.mock_evaluator

        # Mock the evaluator's result
        self.mock_evaluator.evaluate_strings.return_value = {
            "score": 1,
            "value": "Y",
            "reasoning": "Relevant information.",
        }

        # Initialize the Relevance class with a mocked model
        self.relevance = Relevance(llm=self.mock_llm_model)
        self.relevance.evaluator = self.mock_evaluator

        # Add cleanups to stop the patches
        self.addCleanup(self.load_evaluator_patcher.stop)
        self.addCleanup(self.llm_patcher.stop)

    def test_compute_with_mock(self):
        contexts = ["The capital of France is Paris."]
        query = "What is the capital of France?"

        result = self.relevance.compute(contexts, query)

        # Assert that the mocked evaluator's return value is as expected
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["value"], "Y")
        self.assertEqual(result["reasoning"], "Relevant information.")


class TestAccuracyWithMockedRetrieverEvaluator(unittest.TestCase):
    def setUp(self):
        patcher = patch(
            "llama_index.core.evaluation.RetrieverEvaluator.from_metric_names"
        )
        mock_retriever_evaluator = patcher.start()

        # Mock the retriever evaluator
        self.mock_retriever_evaluator = MagicMock()
        mock_retriever_evaluator.return_value = self.mock_retriever_evaluator

        # Mock the retriever evaluator's result
        self.mock_retriever_evaluator.evaluate.return_value.metric_vals_dict = {
            "mrr": 1.0,
            "hit_rate": 0.9,
            "precision": 0.8,
            "recall": 0.7,
        }

        # Mock the index and its retriever
        self.mock_index = MagicMock()
        self.mock_index.as_retriever.return_value = MagicMock()

        # Initialize the Accuracy class with the mocked index
        self.accuracy = Accuracy(index=self.mock_index, k=2)

        self.addCleanup(patcher.stop)

    def test_compute_with_mock(self):
        query = "What is the capital of France?"
        expected_ids = ["123", "456"]

        result = self.accuracy.compute(query, expected_ids)

        # Assert that the mocked evaluator's return value is as expected
        self.assertEqual(result["mrr"], 1.0)
        self.assertEqual(result["hit_rate"], 0.9)
        self.assertEqual(result["precision"], 0.8)
        self.assertEqual(result["recall"], 0.7)
