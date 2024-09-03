import unittest

# import pytest

from saga_llm_evaluation.helpers.llm_metrics import (
    GEval,
    GPTScore,
    SelfCheckGPT,
    Relevance,
    Correctness,
    Faithfulness,
    NegativeRejection,
    HallucinationScore,
)
from tests import LLAMA_MODEL

# skip it for github actions, too many resources needed. Test locally
# pytest.skip(allow_module_level=True) #TODO: Reactivate before pushing


@unittest.skip("Skipping TestGEval for now")
class TestGEval(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geval = GEval(model=LLAMA_MODEL)

    def test_init(self):
        with self.assertRaises(AssertionError):
            GEval(LLAMA_MODEL, "1", 1)
            GEval(LLAMA_MODEL, 1, "1")

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
            self.assertTrue(isinstance(score[aspect][0], float))
            self.assertGreaterEqual(score[aspect][0], 0.0)
            scores[pred] = score[aspect][0]

        self.assertGreaterEqual(
            scores["I am very good, thank you! And you?"], scores["Shut up creep!!!"]
        )


@unittest.skip("Skipping TestSelfCheckGPT for now")
class TestSelfCheckGPT(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selfcheckgpt = SelfCheckGPT(model=LLAMA_MODEL, eval_model=LLAMA_MODEL)

    def test_init(self):
        with self.assertRaises(AssertionError):
            SelfCheckGPT(
                model=LLAMA_MODEL,
                eval_model=LLAMA_MODEL,
                eval_model_name_or_path=1,
                eval_model_basename=1,
            )
            SelfCheckGPT(
                model=LLAMA_MODEL,
                eval_model=LLAMA_MODEL,
                eval_model_name_or_path=1,
                eval_model_basename="1",
            )
            SelfCheckGPT(
                model=LLAMA_MODEL,
                eval_model=LLAMA_MODEL,
                eval_model_name_or_path="1",
                eval_model_basename=1,
            )

    def test_bad_arguments(self):
        user_prompt = "What is the capital of France?"
        prediction = "Paris"
        n_samples = 1

        with self.assertRaises(AssertionError):
            self.selfcheckgpt.compute([user_prompt], prediction, n_samples)
            self.selfcheckgpt.compute(user_prompt, [prediction], n_samples)
            self.selfcheckgpt.compute(user_prompt, prediction, "1")
            self.selfcheckgpt.compute(user_prompt, prediction, 1.3)
            self.selfcheckgpt.compute(user_prompt, prediction, -1)
            self.selfcheckgpt.compute(
                user_prompts=user_prompt, predictions=None, n_samples=5
            )
            self.selfcheckgpt.compute(
                user_prompts=None, predictions=prediction, n_samples=5
            )

    def test_compute(self):
        question = "What is the capital of France?"
        preds = ["Paris", "sandwich"]
        n_samples = 5

        scores = {key: 0 for key in preds}
        for pred in preds:
            score = self.selfcheckgpt.compute(question, pred, n_samples)
            self.assertTrue(isinstance(score[0], float))
            self.assertGreaterEqual(score[0], 0.0)
            self.assertLessEqual(score[0], 1.0)
            scores[pred] = score[0]

        self.assertGreaterEqual(scores["Paris"], scores["sandwich"])


@unittest.skip("Skipping TestGPTScore for now")
class TestGPTScore(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gptscore = GPTScore(model=LLAMA_MODEL)

    def test_init(self):
        with self.assertRaises(AssertionError):
            GPTScore(model=LLAMA_MODEL, model_basename=1, model_name_or_path=1)
            GPTScore(model=LLAMA_MODEL, model_basename="1", model_name_or_path=1)
            GPTScore(model=LLAMA_MODEL, model_basename=1, model_name_or_path="1")

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
                "The cat sat on the mat.", "The dog sat on the log.", custom_prompt=2
            )
            self.gptscore.compute(
                "The cat sat on the mat.",
                "The dog sat on the log.",
                custom_prompt="2",
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
            scores[target] = score[aspect][0]
            self.assertTrue(isinstance(score[aspect][0], float))
            self.assertGreaterEqual(score[aspect][0], 0.0)

        self.assertGreaterEqual(
            scores["I am very fine. Thanks! What about you?"],
            scores["Shut up creep I don't want to talk to you!!!"],
        )


class TestRelevance(unittest.TestCase):
    def test_relevant(self):
        relevance = Relevance()

        query = "Which family do cats belong to?"
        prediction = ["Cats are animals that are apart of the Felidae family."]

        # Act
        result = relevance.compute(query, prediction)

        # Assert
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["value"], "Y")

    def test_irrelevant(self):
        relevance = Relevance()

        query = "Do I need a driving license to drive a car?"
        prediction = [
            "Cats are animals that are apart of the Felidae family. They are often kept as pets and are carnivorous."
        ]

        # Act
        result = relevance.compute(query, prediction)

        # Assert
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["value"], "N")


class TestCorrectness(unittest.TestCase):
    def test_correct(self):
        correctness = Correctness()
        query = "I want to know more about cats."
        references = [
            "Cats are animals of the Felidae family.",
            "Cats are often kept as pets.",
            "Cats are carnivorous mammals.",
        ]

        pred = "Of course! Cats are animals of the Felidae family that are often kept as pets and are carnivorous."
        # Act
        result = correctness.compute(query, pred, references)
        # Assert
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["value"], "Y")

    def test_partially_correct(self):
        correctness = Correctness()
        query = "I want to know more about cats."
        references = [
            "Cats are animals of the Felidae family.",
            "Cats are often kept as pets.",
            "Cats are carnivorous mammals.",
        ]

        pred = "Of course! Cats are animals of the Gasterosteidae family that are often kept as pets and are carnivorous."
        # Act
        result = correctness.compute(query, pred, references)
        # Assert
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["value"], "N")

    def test_incorrect(self):
        correctness = Correctness()
        query = "I want to know more about cats."
        references = [
            "Cats are animals of the Felidae family.",
            "Cats are often kept as pets.",
            "Cats are carnivorous mammals.",
        ]

        pred = "If you want to drive a car, you need a driving license."
        # Act
        result = correctness.compute(query, pred, references)

        # Assert
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["value"], "N")


class TestFaithfulness(unittest.TestCase):
    def test_faithful(self):
        faithfulness = Faithfulness()

        user_prompts = ["Tell me about the Felidae family."]
        references = [
            "Cats are animals of the Felidae family.",
            "Cats are often kept as pets.",
        ]
        pred = "Cats are animals of the Felidae family and are often kept as pets."

        # Act
        result = faithfulness.compute(user_prompts, [pred], references)

        # Assert
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["value"], "Y")

    def test_with_hallucination(self):
        faithfulness = Faithfulness()

        user_prompts = ["Tell me about the Felidae family."]
        references = ["Cats are animals of the Felidae family."]
        pred = (
            "Cats are animals of the Felidae family, and they have wings and can fly."
        )

        # Act
        result = faithfulness.compute(user_prompts, [pred], references)

        # Assert
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["value"], "N")

    def test_unrelated_content(self):
        faithfulness = Faithfulness()

        user_prompts = ["Tell me about the Felidae family."]
        references = ["Cats are animals of the Felidae family."]
        pred = "Driving a car requires a driving license."

        # Act
        result = faithfulness.compute(user_prompts, [pred], references)

        # Assert
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["value"], "N")

    def test_partial_faithfulness(self):
        faithfulness = Faithfulness()

        user_prompts = ["Tell me about the Felidae family."]
        references = [
            "Cats are animals of the Felidae family.",
            "Cats are often kept as pets.",
        ]
        pred = "Cats are part of the Felidae family but they are not kept as pets."

        # Act
        result = faithfulness.compute(user_prompts, [pred], references)

        # Assert
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["value"], "N")

    def test_faithful_but_paraphrased(self):
        faithfulness = Faithfulness()

        user_prompts = ["Tell me about the Felidae family."]
        references = [
            "Cats are animals of the Felidae family.",
            "Cats are often kept as pets.",
        ]
        pred = "Cats belong to the Felidae family and are usually kept as pets."

        # Act
        result = faithfulness.compute(user_prompts, [pred], references)

        # Assert
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["value"], "Y")


class TestNegativeRejection(unittest.TestCase):
    def test_rejection_no_evidence(self):
        negative_rejection = NegativeRejection()

        user_prompts = ["Tell me about unicorns."]
        references = ["Centaurs are mythical creatures."]  # Nothing about unicorns
        pred = "I'm sorry, I cannot provide information on that topic."

        # Act
        result = negative_rejection.compute(user_prompts, [pred], references)

        # Assert
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["value"], "Y")

    def test_no_rejection_evidence(self):
        negative_rejection = NegativeRejection()

        user_prompts = ["Tell me about cats."]
        references = ["Cats are animals of the Felidae family."]  # Evidence is present
        pred = "I'm sorry, I cannot provide information on that topic."

        # Act
        result = negative_rejection.compute(user_prompts, [pred], references)

        # Assert
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["value"], "N")

    def test_rejection_with_nothing(self):
        negative_rejection = NegativeRejection()

        user_prompts = ["How can I apply to engineering in ULiège?"]
        references = [""]
        pred = "I'm sorry, I cannot provide information on that topic."

        # Act
        result = negative_rejection.compute(user_prompts, [pred], references)

        # Assert
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["value"], "Y")

    def test_no_rejection_correct_info(self):
        negative_rejection = NegativeRejection()

        user_prompts = ["Tell me about dogs."]
        references = ["Dogs are domesticated mammals, not natural wild animals."]
        pred = "I'm sorry, I cannot provide information on that topic."

        # Act
        result = negative_rejection.compute(user_prompts, [pred], references)

        # Assert
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["value"], "N")


class TestHallucinationScore(unittest.TestCase):
    def setUp(self):
        self.hallucination_detector = HallucinationScore()

    def test_exact_match(self):
        predictions = ["Cats are animals of the Felidae family."]
        references = ["Cats are animals of the Felidae family."]

        result = self.hallucination_detector.compute(predictions, references)

        self.assertEqual(result["f1_score"], 1.0)
        self.assertEqual(result["exact_match"], 1)

    def test_partial_hallucination(self):
        predictions = ["Cats are animals of the Felidae family and they can fly."]
        references = ["Cats are animals of the Felidae family."]

        result = self.hallucination_detector.compute(predictions, references)

        self.assertGreater(result["f1_score"], 0.5)
        self.assertLess(result["f1_score"], 1.0)
        self.assertEqual(result["exact_match"], 0)

    def test_no_reference(self):
        predictions = ["Cats are animals of the Felidae family."]
        references = [""]

        result = self.hallucination_detector.compute(predictions, references)

        self.assertEqual(result["f1_score"], 0.0)
        self.assertEqual(result["exact_match"], 0)

    def test_mixed_predictions(self):
        predictions = [
            "Cats are animals of the Felidae family and they can fly.",
            "Dogs are animals of the Canidae family.",
        ]
        references = [
            "Cats are animals of the Felidae family.",
            "Dogs are animals of the Canidae family.",
        ]

        result = self.hallucination_detector.compute(predictions, references)

        self.assertGreater(result["f1_score"], 0.5)
        self.assertEqual(result["exact_match"], 0)
