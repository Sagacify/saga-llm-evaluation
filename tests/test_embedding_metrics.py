import unittest

from sagacify_llm_evaluation.helpers.embedding_metrics import MAUVE, BERTScore


class TestBERTScore(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bertscore = BERTScore()

    def test_compute(self):
        """Tests that the BERTScore class computes the correct scores. And that the scores are the same when the same inputs are given."""
        references = ["The cat sat on the mat.", "The dog sat on the log."]
        predictions = ["The cat sat on the mat.", "The dog sat on the log."]

        scores = self.bertscore.compute(references, predictions)
        self.assertEqual(len(scores["precision"]), len(references))
        self.assertEqual(len(scores["recall"]), len(references))
        self.assertEqual(len(scores["f1"]), len(references))

        scores_2 = self.bertscore.compute(references, predictions)
        self.assertEqual(scores["precision"], scores_2["precision"])
        self.assertEqual(scores["recall"], scores_2["recall"])
        self.assertEqual(scores["f1"], scores_2["f1"])

    def test_compute_improved_input(self):
        """Tests that the BERTScore improves for a better prediction."""
        reference = "The cat sat on the mat."
        prediction = "The dog sat on the mat."
        better_prediction = "The cat sat on the mat."

        scores = self.bertscore.compute([reference], [prediction])
        better_scores = self.bertscore.compute([reference], [better_prediction])

        self.assertGreater(better_scores["f1"][0], scores["f1"][0])


class TestMAUVE(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mauve = MAUVE()

    def test_compute(self):
        """Tests that the MAUVE class computes the same scores when the same inputs are given."""
        references = ["The cat sat on the mat.", "The dog sat on the log."]
        predictions = ["The cat sat on the mat.", "The dog sat on the log."]
        scores = self.mauve.compute(references, predictions)
        scores_2 = self.mauve.compute(references, predictions)
        self.assertEqual(scores.mauve, scores_2.mauve)

    def test_compute_improved_input(self):
        """Tests that the MAUVE Score improves for a better prediction."""
        reference = "The cat sat on the mat."
        prediction = "The dog sat on the mat."
        better_prediction = "The cat sat on the mat."

        scores = self.mauve.compute([reference], [prediction])
        better_scores = self.mauve.compute([reference], [better_prediction])

        self.assertGreater(better_scores.mauve, scores.mauve)
