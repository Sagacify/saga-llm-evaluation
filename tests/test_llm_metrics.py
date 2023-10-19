import unittest

from saga_llm_evaluation_ml.model.helpers.llm_metrics import GPTScore


class TestGPTScore(unittest.TestCase):
    def test_bad_arguments(self):
        gptscore = GPTScore()

        with self.assertRaises(AssertionError):
            gptscore.compute(["The cat sat on the mat."], ["The dog sat on the log."])
            gptscore.compute(
                "The cat sat on the mat.", "The dog sat on the log.", model="random"
            )
            gptscore.compute(
                "The cat sat on the mat.",
                "The dog sat on the log.",
                model="meta-llama/Llama-2-7b-chat-hf",
                prompt=10,
            )
            gptscore.compute(
                "The cat sat on the mat.",
                "The dog sat on the log.",
                prompt="Summarize",
                a="ERR",
            )
            gptscore.compute(
                "The cat sat on the mat.",
                "The dog sat on the log.",
                prompt="Summarize",
                d="summ",
            )
            gptscore.compute(
                "The cat sat on the mat.",
                "The dog sat on the log.",
                prompt="Summarize",
                a="ERR",
                d="summ",
            )
            gptscore.compute(
                "The cat sat on the mat.", "The dog sat on the log.", a="ERR", d=None
            )
            gptscore.compute(
                "The cat sat on the mat.", "The dog sat on the log.", a=None, d="summ"
            )
            gptscore.compute(
                "The cat sat on the mat.", "The dog sat on the log.", a=2, d="summ"
            )
            gptscore.compute(
                "The cat sat on the mat.", "The dog sat on the log.", a="ERR", d=None
            )
            gptscore.compute(
                "The cat sat on the mat.",
                "The dog sat on the log.",
                a="notvalid",
                d="summ",
            )
            gptscore.compute(
                "The cat sat on the mat.", "The dog sat on the log.", a="ERR", d="D2T"
            )

    def test_compute(self):
        """Tests that the GPTScore computes a higher score for a better prediction."""
        source = "State something true."
        pred = "The cat eats elephants."
        better_pred = "The cat eats mice."

        gptscore = GPTScore()
        score = gptscore.compute(source, pred, a="ERR", d="diag")
        score_2 = gptscore.compute(source, better_pred, a="ERR", d="diag")
        self.assertGreater(score_2, score)
