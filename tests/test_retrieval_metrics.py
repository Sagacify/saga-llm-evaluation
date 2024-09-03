import unittest

from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex

from saga_llm_evaluation.helpers.retrieval_metrics import Relevance, Accuracy


class TestRelevance(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relevance = Relevance()

    def test_compute(self):
        relevance = Relevance()

        query = "Cats are animals of the Felidae family."
        retrieved_info = [
            "Cats are animals that are apart of the Felidae family.\
                          They are often kept as pets and are carnivorous."
        ]

        # Act
        result = relevance.compute(retrieved_info, query)

        # Assert
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["value"], "Y")

        # Change the query to sth that is not exactly in the index
        query = "elephants"
        # Act
        result = relevance.compute(retrieved_info, query)

        # Assert
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["value"], "N")


class TestAccuracy(unittest.TestCase):
    def test_accuracy_with_real_data(self):
        # Create a real index and populate it with data
        node1 = TextNode(text="Cats are animals of the Felidae family.", id_="node_id1")
        node2 = TextNode(text="Cats are often kept as pets.", id_="node_id2")
        node3 = TextNode(text="Dogs are animals of the Canidae family.", id_="node_id3")
        node4 = TextNode(text="Dolphins are marine mammals.", id_="node_id4")
        nodes = [node1, node2, node3, node4]
        index = VectorStoreIndex(nodes)

        accuracy = Accuracy(index=index, k=2)

        query = "cats"

        # Act
        result = accuracy.compute(query)

        # Assert
        self.assertEqual(result["mrr"], 1.0)
        self.assertEqual(result["hit_rate"], 1.0)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)

        # Change the query to sth that is not exactly in the index
        query = "elephants"

        # Act
        result = accuracy.compute(query)

        # Assert that score is between 0 and 1
        self.assertGreaterEqual(result["mrr"], 0.0)
        self.assertGreaterEqual(result["hit_rate"], 0.0)
        self.assertLessEqual(result["mrr"], 1.0)
        self.assertLessEqual(result["hit_rate"], 1.0)
        self.assertGreaterEqual(result["precision"], 0.0)
        self.assertGreaterEqual(result["recall"], 0.0)
        self.assertLessEqual(result["precision"], 1.0)
        self.assertLessEqual(result["recall"], 1.0)