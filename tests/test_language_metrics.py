import unittest

from saga_llm_evaluation_ml.helpers.language_metrics import BLEURTScore, QSquared


class TestBLEURTScore(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bleurt = BLEURTScore()

    def test_compute(self):
        """Tests that the BLEURTScore class computes the correct scores. And that the scores are the same when the same inputs are given."""
        references = ["The cat sat on the mat.", "The dog sat on the log."]
        predictions = ["The cat sat on the mat.", "The dog sat on the log."]

        scores = self.bleurt.compute(references, predictions)
        scores_2 = self.bleurt.compute(references, predictions)
        self.assertEqual(scores, scores_2)

    def test_compute_improved_input(self):
        """Tests that the BLEURTScore improves for a better prediction."""
        reference = "The cat sat on the mat."
        prediction = "The dog sat on the mat."
        better_prediction = "The cat sat on the mat."

        scores = self.bleurt.compute([reference], [prediction])
        better_scores = self.bleurt.compute([reference], [better_prediction])
        self.assertGreater(better_scores["scores"], scores["scores"])


class TestQSquared(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_squared = QSquared()

    def test_compute(self):
        """Tests that the QSquared class computes the correct scores. And that the scores are the same when the same inputs are given."""
        knowledges = [
            "The beautiful cat sat on the ugly mat.",
            "The ugly dog sat on the beautiful mat.",
        ]
        predictions = ["The cat sat on the mat.", "The dog sat on the mat."]

        for know, pred in zip(knowledges, predictions):
            scores = self.q_squared.compute([pred], [know], single=True)
            scores_2 = self.q_squared.compute([pred], [know], single=True)
            self.assertEqual(scores, scores_2)

    def test_compute_improved_input(self):
        """Tests that the QSquared improves for a better prediction."""
        knowledge = "Chronic urethral obstruction because of urinary calculi, prostatic hyperophy, tumors, normal pregnancy, tumors, uterine prolapse or functional disorders cause hydronephrosis which by definition is used to describe dilatation of renal pelvis and calculus associated with progressive atrophy of the kidney due to obstruction to the outflow of urine Refer Robbins 7yh/9,1012,9/e. P950"
        prediction = "The cat sat on the mat."
        better_prediction = "Chronic urethral obstruction due to benign prismatic hyperplasia can lead to hyperophy"

        scores = self.q_squared.compute([prediction], [knowledge], single=True)
        better_scores = self.q_squared.compute(
            [better_prediction], [knowledge], single=True
        )
        self.assertGreater(better_scores["avg_f1"][0], scores["avg_f1"][0])
