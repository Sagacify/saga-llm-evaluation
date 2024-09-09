import unittest

from unittest.mock import patch, MagicMock

from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex

from saga_llm_evaluation.helpers.retrieval_metrics import Relevance, Accuracy

from tests import LANGCHAIN_GPT_MODEL as MODEL


@unittest.skip("Skip TestRelevance")
class TestRelevance(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relevance = Relevance(llm=MODEL)

    def test_relevant(self):
        query = "Cats are animals of the Felidae family."
        retrieved_info = [
            "Cats are animals that are apart of the Felidae family.\
                          They are often kept as pets and are carnivorous."
        ]

        # Act
        result = self.relevance.compute(retrieved_info, query)

        # Assert
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["value"], "Y")

    def test_irrelevant(self):
        query = "elephants"
        retrieved_info = [
            "Cats are animals that are apart of the Felidae family.\
                          They are often kept as pets and are carnivorous."
        ]

        # Act
        result = self.relevance.compute(retrieved_info, query)

        # Assert
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["value"], "N")


@unittest.skip("Skip TestAccuracy")
class TestAccuracy(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create a real index and populate it with data
        node1 = TextNode(text="Cats are animals of the Felidae family.", id_="node_id1")
        node2 = TextNode(text="Cats are often kept as pets.", id_="node_id2")
        node3 = TextNode(text="Dogs are animals of the Canidae family.", id_="node_id3")
        node4 = TextNode(text="Dolphins are marine mammals.", id_="node_id4")

        self.index = VectorStoreIndex([node1, node2, node3, node4])
        self.accuracy = Accuracy(index=self.index, k=2)

    def test_accuracy_with_real_data(self):
        query = "cats"

        # Act
        result = self.accuracy.compute(query, expected_ids=["node_id1", "node_id2"])

        # Assert
        self.assertEqual(result["mrr"], 1.0)
        self.assertEqual(result["hit_rate"], 1.0)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)

        # Change the query to sth that is not exactly in the index
        query = "elephants"

        # Act
        result = self.accuracy.compute(query, expected_ids=["node_id1", "node_id2"])

        # Assert that score is between 0 and 1
        self.assertGreaterEqual(result["mrr"], 0.0)
        self.assertGreaterEqual(result["hit_rate"], 0.0)
        self.assertLessEqual(result["mrr"], 1.0)
        self.assertLessEqual(result["hit_rate"], 1.0)
        self.assertGreaterEqual(result["precision"], 0.0)
        self.assertGreaterEqual(result["recall"], 0.0)
        self.assertLessEqual(result["precision"], 1.0)
        self.assertLessEqual(result["recall"], 1.0)


class TestRelevanceWithMockedEvaluator(unittest.TestCase):
    def setUp(self):
        patcher = patch("langchain.evaluation.load_evaluator")
        mock_load_evaluator = patcher.start()

        # Mock the evaluator
        self.mock_evaluator = MagicMock()
        mock_load_evaluator.return_value = self.mock_evaluator

        # Mock the evaluator's result
        self.mock_evaluator.evaluate_strings.return_value = {
            "score": 1,
            "value": "Y",
            "reasoning": "Relevant information.",
        }

        # Initialize the Relevance class with a mocked model (None means no LLM is provided)
        self.relevance = Relevance()
        self.relevance.evaluator = self.mock_evaluator

        self.addCleanup(patcher.stop)

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
