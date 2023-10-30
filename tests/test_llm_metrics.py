import unittest

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from saga_llm_evaluation_ml.helpers.llm_metrics import GEval, GPTScore, SelfCheckGPT


class TestGEval(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geval = GEval()

    def test_init(self):
        with self.assertRaises(AssertionError):
            GEval(1, 1)
            GEval("1", 1)
            GEval(1, "1")

    def test_bad_arguments(self):
        source = "Hi how are you"
        pred = "Im ok"
        task = "diag"
        aspect = "ENG"

        with self.assertRaises(AssertionError):
            self.geval.compute([source], pred, task, aspect)
            self.geval.compute(source, [pred], task, aspect)
            self.geval.compute(source, pred, 1, aspect)
            self.geval.compute(source, pred, task, 1)
            self.geval.compute(source, pred, task, "notvalid")
            self.geval.compute(source, pred, "notvalid", aspect)
            self.geval.compute(source, pred, task, aspect=None)
            self.geval.compute(source, pred, task=None, aspect=aspect)

    def test_compute(self):
        source = "Hi how are you?"
        preds = ["Shut up creep!!!", "I am very good, thank you! And you?"]
        task = "diag"
        aspect = "POL"

        scores = {key: 0 for key in preds}
        for pred in preds:
            score = self.geval.compute(source, pred, task, aspect)
            self.assertTrue(isinstance(score, float))
            self.assertGreaterEqual(score, 0.0)
            scores[pred] = score

        self.assertGreaterEqual(
            scores["I am very good, thank you! And you?"], scores["Shut up creep!!!"]
        )


class TestSelfCheckGPT(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = hf_hub_download(
            repo_id="TheBloke/Llama-2-7b-Chat-GGUF",
            filename="llama-2-7b-chat.Q4_K_M.gguf",  # the model is in bin format
        )
        self.model = Llama(model_path=self.model_path, n_threads=2, verbose=False)
        self.selfcheckgpt = SelfCheckGPT(self.model)

    def test_init(self):
        with self.assertRaises(AssertionError):
            SelfCheckGPT(self.model, eval_model_name_or_path=1, eval_model_basename=1)
            SelfCheckGPT(self.model, eval_model_name_or_path=1, eval_model_basename="1")
            SelfCheckGPT(self.model, eval_model_name_or_path="1", eval_model_basename=1)

    def test_bad_arguments(self):
        question = "What is the capital of France?"
        pred = "Paris"
        n_samples = 1

        with self.assertRaises(AssertionError):
            self.selfcheckgpt.compute([question], pred, n_samples)
            self.selfcheckgpt.compute(question, [pred], n_samples)
            self.selfcheckgpt.compute(question, pred, "1")
            self.selfcheckgpt.compute(question, pred, 1.0)
            self.selfcheckgpt.compute(question, pred, -1)
            self.selfcheckgpt.compute(question=question, pred=None, n_samples=5)
            self.selfcheckgpt.compute(question=None, pred=pred, n_samples=5)

    def test_compute(self):
        question = "What is the capital of France?"
        preds = ["Paris", "sandwich"]
        n_samples = 10

        scores = {key: 0 for key in preds}
        for pred in preds:
            score = self.selfcheckgpt.compute(question, pred, n_samples)
            self.assertTrue(isinstance(score, float))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            scores[pred] = score

        self.assertGreaterEqual(scores["Paris"], scores["sandwich"])


class TestGPTScore(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gptscore = GPTScore()

    def test_init(self):
        with self.assertRaises(AssertionError):
            GPTScore(model_basename=1, model_name_or_path=1)
            GPTScore(model_basename="1", model_name_or_path=1)
            GPTScore(model_basename=1, model_name_or_path="1")

    def test_bad_arguments(self):

        with self.assertRaises(AssertionError):
            self.gptscore.compute(
                ["The cat sat on the mat."], ["The dog sat on the log."]
            )
            self.gptscore.compute(
                "The cat sat on the mat.", ["The dog sat on the log."]
            )
            self.gptscore.compute("The cat sat on the mat.", "The dog sat on the log.")
            self.gptscore.compute(
                "The cat sat on the mat.", "The dog sat on the log.", prompt=2
            )
            self.gptscore.compute(
                "The cat sat on the mat.",
                "The dog sat on the log.",
                prompt="2",
                aspect="COV",
                task="diag",
            )
            self.gptscore.compute(
                "The cat sat on the mat.",
                "The dog sat on the log.",
                aspect=2,
                task="diag",
            )
            self.gptscore.compute(
                "The cat sat on the mat.",
                "The dog sat on the log.",
                aspect="COV",
                task=2,
            )
            self.gptscore.compute(
                "The cat sat on the mat.",
                "The dog sat on the log.",
                aspect="COV",
                task="notvalid",
            )
            self.gptscore.compute(
                "The cat sat on the mat.",
                "The dog sat on the log.",
                aspect="notvalid",
                task="diag",
            )
            self.gptscore.compute(
                "The cat sat on the mat.", "The dog sat on the log.", aspect="COV"
            )
            self.gptscore.compute(
                "The cat sat on the mat.", "The dog sat on the log.", task="diag"
            )

    def test_compute(self):
        source = "Hi how are you?"
        preds = [
            "I am very fine. Thanks! What about you?",
            "Shut up creep I don't want to talk to you!!!",
        ]
        # prompt = "Task: evaluate how polite this dialog is."
        aspect = "LIK"
        task = "diag"

        scores = {key: 0 for key in preds}
        for target in preds:
            score = self.gptscore.compute(source, target, aspect=aspect, task=task)
            scores[target] = score
            self.assertTrue(isinstance(score, float))
            self.assertGreaterEqual(score, 0.0)

        self.assertGreaterEqual(
            scores["I am very fine. Thanks! What about you?"],
            scores["Shut up creep I don't want to talk to you!!!"],
        )
