import unittest

from saga_llm_evaluation_ml.model.helpers.language_metrics import BLEURTScore


class TestBLEURTScore(unittest.TestCase):
    def test_compute(self):
        """Tests that the BLEURTScore class computes the correct scores. And that the scores are the same when the same inputs are given."""
        references = ["The cat sat on the mat.", "The dog sat on the log."]
        predictions = ["The cat sat on the mat.", "The dog sat on the log."]

        bleurt = BLEURTScore()

        scores = bleurt.compute(references, predictions)
        scores_2 = bleurt.compute(references, predictions)
        self.assertEqual(scores, scores_2)

    def test_compute_improved_input(self):
        """Tests that the BLEURTScore improves for a better prediction."""
        reference = "The cat sat on the mat."
        prediction = "The dog sat on the mat."
        better_prediction = "The cat sat on the mat."

        bleurt = BLEURTScore()

        scores = bleurt.compute([reference], [prediction])
        better_scores = bleurt.compute([reference], [better_prediction])
        self.assertGreater(better_scores["scores"], scores["scores"])
