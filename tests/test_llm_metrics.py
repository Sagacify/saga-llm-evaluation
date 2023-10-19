import unittest

from saga_llm_evaluation_ml.helpers.llm_metrics import GPTScore


class TestGPTScore(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(AssertionError):
            GPTScore(model=100)
            GPTScore(model="notvalid")

    def test_bad_arguments(self):
        gptscore = GPTScore()

        with self.assertRaises(AssertionError):
            gptscore.compute("The cat sat on the mat.", "The dog sat on the log.")
            gptscore.compute(
                ["The cat sat on the mat."],
                ["The dog sat on the log."],
                prompts=10,
            )
            gptscore.compute(
                ["The cat sat on the mat."],
                ["The dog sat on the log."],
                prompts="Summarize",
                aspect="ERR",
            )
            gptscore.compute(
                ["The cat sat on the mat."],
                ["The dog sat on the log."],
                prompts="Summarize",
                task="summ",
            )
            gptscore.compute(
                ["The cat sat on the mat."],
                ["The dog sat on the log."],
                prompts="Summarize",
                aspect="ERR",
                task="summ",
            )
            gptscore.compute(
                ["The cat sat on the mat."],
                ["The dog sat on the log."],
                aspect="ERR",
                task=None,
            )
            gptscore.compute(
                ["The cat sat on the mat."],
                ["The dog sat on the log."],
                aspect=None,
                task="summ",
            )
            gptscore.compute(
                ["The cat sat on the mat."],
                ["The dog sat on the log."],
                aspect=2,
                task="summ",
            )
            gptscore.compute(
                ["The cat sat on the mat."],
                ["The dog sat on the log."],
                aspect="ERR",
                task=None,
            )
            gptscore.compute(
                ["The cat sat on the mat."],
                ["The dog sat on the log."],
                aspect="notvalid",
                task="summ",
            )
            gptscore.compute(
                ["The cat sat on the mat."],
                ["The dog sat on the log."],
                aspect="ERR",
                task="D2T",
            )

    def test_compute_gpt2(self):
        """Tests that the GPTScore computes a higher score for a better prediction with gpt2."""
        sources = ["State something true.", "State something true."]
        preds = ["The cat eats elephants.", "The cat eats mice."]

        gptscore = GPTScore()

        # gpt2
        scores = gptscore.compute(sources, preds, aspect="ERR", task="diag")
        self.assertGreater(scores[1], scores[0])

    # def test_compute_mistral(self):
    #     """
    #     Tests that the GPTScore computes a higher score for a better prediction
    #     with mistralai/Mistral-7B-v0.1.
    #     """
    #     source = "State something true."
    #     pred = "The cat eats elephants."
    #     better_pred = "The cat eats mice."

    #     gptscore = GPTScore()

    #     # mistralai/Mistral-7B-v0.1
    #     score = gptscore.compute(source, pred, aspect="ERR", task="diag", model="mistralai/Mistral-7B-v0.1")
    #     score_2 = gptscore.compute(source, better_pred, aspect="ERR", task="diag", model="mistralai/Mistral-7B-v0.1")
    #     self.assertGreater(score_2, score)

    # def test_compute_llama(self):
    #     """
    #     Tests that the GPTScore computes a higher score for a better prediction
    #     with meta-llama/Llama-2-7b-chat-hf.
    #     """
    #     source = "State something true."
    #     pred = "The cat eats elephants."
    #     better_pred = "The cat eats mice."

    #     gptscore = GPTScore()

    #     # meta-llama/Llama-2-7b-chat-hf
    #     score = gptscore.compute(source, pred, aspect="ERR", task="diag", model="meta-llama/Llama-2-7b-chat-hf")
    #     score_2 = gptscore.compute(source, better_pred, aspect="ERR", task="diag",
    #       model="meta-llama/Llama-2-7b-chat-hf")
    #     self.assertGreater(score_2, score)
