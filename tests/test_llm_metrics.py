import unittest

from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from saga_llm_evaluation_ml.helpers.llm_metrics import GPTScore, GEval, SelfCheckGPT


class TestGEval(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(AssertionError):
            GEval(1, 1)
            GEval("1", 1)
            GEval(1, "1")

    def test_bad_arguments(self):
        geval = GEval()

        source = "Hi how are you"
        pred = "Im ok"
        task = "diag"
        aspect = "ENG"

        with self.assertRaises(AssertionError):
            geval.compute([source], pred, task, aspect)
            geval.compute(source, [pred], task, aspect)
            geval.compute(source, pred, 1, aspect)
            geval.compute(source, pred, task, 1)
            geval.compute(source, pred, task, "notvalid")
            geval.compute(source, pred, "notvalid", aspect)
            geval.compute(source, pred, task, aspect=None)
            geval.compute(source, pred, task=None, aspect=aspect)

    def test_compute(self):
        geval = GEval()

        source = "Hi how are you?"
        preds = ["Shut up creep!!!", "I am very good, thank you! And you?"]
        task = "diag"
        aspect = "POL"

        scores = {key: 0 for key in preds}
        for pred in preds:
            score = geval.compute(source, pred, task, aspect)
            self.assertTrue(isinstance(score, float))
            self.assertGreaterEqual(score, 0.0)
            scores[pred] = score

        self.assertGreaterEqual(
            scores["I am very good, thank you! And you?"], scores["Shut up creep!!!"]
        )


class TestSelfCheckGPT(unittest.TestCase):
    def test_init(self):
        model_name_or_path = "TheBloke/Llama-2-7b-Chat-GGUF"
        model_basename = "llama-2-7b-chat.Q4_K_M.gguf"  # the model is in bin format

        model_path = hf_hub_download(
            repo_id=model_name_or_path, filename=model_basename
        )
        model = Llama(model_path=model_path, n_threads=2, verbose=False)  # CPU cores

        with self.assertRaises(AssertionError):
            SelfCheckGPT(model, eval_model_name_or_path=1, eval_model_basename=1)
            SelfCheckGPT(model, eval_model_name_or_path=1, eval_model_basename="1")
            SelfCheckGPT(model, eval_model_name_or_path="1", eval_model_basename=1)

    def test_bad_arguments(self):

        model_name_or_path = "TheBloke/Llama-2-7b-Chat-GGUF"
        model_basename = "llama-2-7b-chat.Q4_K_M.gguf"  # the model is in bin format

        model_path = hf_hub_download(
            repo_id=model_name_or_path, filename=model_basename
        )
        model = Llama(model_path=model_path, n_threads=2, verbose=False)  # CPU cores

        selfcheckgpt = SelfCheckGPT(model)
        question = "What is the capital of France?"
        pred = "Paris"
        n_samples = 1

        with self.assertRaises(AssertionError):
            selfcheckgpt.compute([question], pred, n_samples)
            selfcheckgpt.compute(question, [pred], n_samples)
            selfcheckgpt.compute(question, pred, "1")
            selfcheckgpt.compute(question, pred, 1.0)
            selfcheckgpt.compute(question, pred, -1)
            selfcheckgpt.compute(question=question, pred=None, n_samples=5)
            selfcheckgpt.compute(question=None, pred=pred, n_samples=5)

    def test_compute(self):
        model_name_or_path = "TheBloke/Llama-2-7b-Chat-GGUF"
        model_basename = "llama-2-7b-chat.Q4_K_M.gguf"

        model_path = hf_hub_download(
            repo_id=model_name_or_path, filename=model_basename
        )
        model = Llama(model_path=model_path, n_threads=2, verbose=False)  # CPU cores

        selfcheckgpt = SelfCheckGPT(model)
        question = "What is the capital of France?"
        preds = ["Paris", "sandwich"]
        n_samples = 10

        scores = {key: 0 for key in preds}
        for pred in preds:
            score = selfcheckgpt.compute(question, pred, n_samples)
            self.assertTrue(isinstance(score, float))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            scores[pred] = score

        self.assertGreaterEqual(scores["Paris"], scores["sandwich"])


class TestGPTScore(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(AssertionError):
            GPTScore(model_basename=1, model_name_or_path=1)
            GPTScore(model_basename="1", model_name_or_path=1)
            GPTScore(model_basename=1, model_name_or_path="1")

    def test_bad_arguments(self):
        gptscore = GPTScore()

        with self.assertRaises(AssertionError):
            gptscore.compute(["The cat sat on the mat."], ["The dog sat on the log."])
            gptscore.compute("The cat sat on the mat.", ["The dog sat on the log."])
            gptscore.compute("The cat sat on the mat.", "The dog sat on the log.")
            gptscore.compute(
                "The cat sat on the mat.", "The dog sat on the log.", prompt=2
            )
            gptscore.compute(
                "The cat sat on the mat.",
                "The dog sat on the log.",
                prompt="2",
                aspect="COV",
                task="diag",
            )
            gptscore.compute(
                "The cat sat on the mat.",
                "The dog sat on the log.",
                aspect=2,
                task="diag",
            )
            gptscore.compute(
                "The cat sat on the mat.",
                "The dog sat on the log.",
                aspect="COV",
                task=2,
            )
            gptscore.compute(
                "The cat sat on the mat.",
                "The dog sat on the log.",
                aspect="COV",
                task="notvalid",
            )
            gptscore.compute(
                "The cat sat on the mat.",
                "The dog sat on the log.",
                aspect="notvalid",
                task="diag",
            )
            gptscore.compute(
                "The cat sat on the mat.", "The dog sat on the log.", aspect="COV"
            )
            gptscore.compute(
                "The cat sat on the mat.", "The dog sat on the log.", task="diag"
            )

    def test_compute(self):
        gptscore = GPTScore()

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
            score = gptscore.compute(source, target, aspect=aspect, task=task)
            scores[target] = score
            self.assertTrue(isinstance(score, float))
            self.assertGreaterEqual(score, 0.0)

        self.assertGreaterEqual(
            scores["I am very fine. Thanks! What about you?"],
            scores["Shut up creep I don't want to talk to you!!!"],
        )
