import unittest
from unittest.mock import MagicMock, patch
from langchain_core.language_models.chat_models import BaseChatModel

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

from tests import LANGCHAIN_GPT_MODEL as MODEL


@unittest.skip("Skip TestGEval")
class TestGEval(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geval = GEval(model=MODEL)
        self.geval_no_model = GEval()

    def test_init(self):
        with self.assertRaises(AssertionError):
            GEval(1)
            GEval("model")

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

    def test_compute_no_model(self):
        source = "Hi how are you?"
        preds = ["Shut up creep!!!", "I am very good, thank you! And you?"]
        task = "diag"
        aspect = "POL"

        scores = {key: 0 for key in preds}
        for pred in preds:
            score = self.geval_no_model.compute(source, pred, task, aspect)
            self.assertTrue(isinstance(score[aspect][0], float))
            self.assertGreaterEqual(score[aspect][0], 0.0)
            scores[pred] = score[aspect][0]

        self.assertGreaterEqual(
            scores["I am very good, thank you! And you?"], scores["Shut up creep!!!"]
        )


@unittest.skip("Skip TestSelfCheckGPT")
class TestSelfCheckGPT(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selfcheckgpt = SelfCheckGPT(model=MODEL, eval_model=MODEL)
        self.selfcheckgpt_no_eval_model = SelfCheckGPT(model=MODEL)

    def test_init(self):
        with self.assertRaises(AssertionError):
            SelfCheckGPT(
                model=None,
            )
            SelfCheckGPT(
                model=1,
            )
            SelfCheckGPT(
                model="test",
            )
            SelfCheckGPT(
                model=MODEL,
                eval_model="test",
            )
            SelfCheckGPT(
                model=MODEL,
                eval_model=1,
            )
            SelfCheckGPT(
                model="lol",
                eval_model=1,
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

    def test_compute_eval_model_given(self):
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

    def test_compute_no_eval_model(self):
        question = "What is the capital of France?"
        preds = ["Paris", "sandwich"]
        n_samples = 5

        scores = {key: 0 for key in preds}
        for pred in preds:
            score = self.selfcheckgpt_no_eval_model.compute(question, pred, n_samples)
            self.assertTrue(isinstance(score[0], float))
            self.assertGreaterEqual(score[0], 0.0)
            self.assertLessEqual(score[0], 1.0)
            scores[pred] = score[0]

        self.assertGreaterEqual(scores["Paris"], scores["sandwich"])


@unittest.skip("Skip TestGPTScore")
class TestGPTScore(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gptscore = GPTScore(model=MODEL)
        self.gptscore_no_model = GPTScore()

    def test_init(self):
        with self.assertRaises(AssertionError):
            GPTScore(model=1)
            GPTScore(model="lol")
            GPTScore(model=[None])

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

    def test_compute_diag(self):
        source = "Hi how are you?"
        preds = [
            "I am very fine. Thanks! What about you?",
            "Shut up creep I don't want to talk to you!!!",
        ]

        aspect = "LIK"
        task = "diag"

        scores = {key: 0 for key in preds}
        for target in preds:
            score = self.gptscore.compute(source, target, aspect=aspect, task=task)
            scores[target] = score[aspect][0]
            self.assertTrue(isinstance(score[aspect][0], float))
            self.assertLessEqual(score[aspect][0], 0.0)

        self.assertGreaterEqual(
            scores["I am very fine. Thanks! What about you?"],
            scores["Shut up creep I don't want to talk to you!!!"],
        )

    def test_compute_no_model_diag(self):
        source = "Hi how are you?"
        sentence1 = "I am very fine. Thanks! What about you?"
        sentence2 = "Shut up creep I don't want to talk to you!!!"
        preds = [sentence1, sentence2]

        aspect = "LIK"
        task = "diag"

        scores = {key: 0 for key in preds}
        for target in preds:
            score = self.gptscore_no_model.compute(
                source, target, aspect=aspect, task=task
            )
            scores[target] = score[aspect][0]
            self.assertTrue(isinstance(score[aspect][0], float))
            self.assertLessEqual(score[aspect][0], 0.0)

        self.assertGreaterEqual(
            scores[sentence1],
            scores[sentence2],
        )

    def test_compute_data2text(self):
        source = str(
            {
                "type": "noodle bar",
                "food type": "chinese",
                "area": "city centre",
                "price": "cheap",
            }
        )
        sentence1 = "There is a Chinese restaurant called Noodle Bar located in the city centre, and it has a moderate price range."
        sentence2 = "I love cats."

        preds = [sentence1, sentence2]
        aspect = "INF"
        task = "data_to_text"

        scores = {key: 0 for key in preds}
        for target in preds:
            score = self.gptscore.compute(source, target, aspect=aspect, task=task)
            scores[target] = score[aspect][0]
            self.assertTrue(isinstance(score[aspect][0], float))
            self.assertLessEqual(score[aspect][0], 0.0)

        self.assertGreaterEqual(
            scores[sentence1],
            scores[sentence2],
        )

    def test_compute_mt(self):
        source = "我喜欢学习新的语言。"
        sentence1 = "I like learning new languages."
        sentence2 = "I am a cat."

        preds = [sentence1, sentence2]
        aspect = "ACC"
        task = "machine_translation"

        scores = {key: 0 for key in preds}
        for target in preds:
            score = self.gptscore.compute(source, target, aspect=aspect, task=task)
            scores[target] = score[aspect][0]
            self.assertTrue(isinstance(score[aspect][0], float))
            self.assertLessEqual(score[aspect][0], 0.0)

        self.assertGreaterEqual(
            scores[sentence1],
            scores[sentence2],
        )

    def test_compute_summary(self):
        source = "The city council has approved a new plan to reduce traffic congestion by adding more bike lanes and increasing public transportation options. The plan, which has been in the works for two years, aims to reduce the number of cars on the road by 20% over the next five years. The mayor said that the plan is crucial to improving the quality of life in the city and reducing pollution levels."
        sentence1 = "The city council approved a plan to reduce traffic by adding bike lanes and improving public transportation, aiming for a 20% reduction in cars over five years."
        sentence2 = "I love cats."

        preds = [sentence1, sentence2]
        aspect = "FAC"
        task = "summ"

        scores = {key: 0 for key in preds}
        for target in preds:
            score = self.gptscore.compute(source, target, aspect=aspect, task=task)
            scores[target] = score[aspect][0]
            self.assertTrue(isinstance(score[aspect][0], float))
            self.assertLessEqual(score[aspect][0], 0.0)

        self.assertGreaterEqual(
            scores[sentence1],
            scores[sentence2],
        )

    def test_add_criterion(self):
        task = "calc"
        code = "COR"
        desc = "The result {pred} of the calculation {src} is correct."

        with self.assertRaises(AssertionError):
            # Wrong arguments
            self.gptscore.add_criterion(task=1, code=2, desc=3)
            self.gptscore.add_criterion(task=task, code=2, desc=3)
            self.gptscore.add_criterion(task=1, code=code, desc=3)
            self.gptscore.add_criterion(task=1, code=2, desc=desc)
            self.gptscore.add_criterion(task=1, code=2, desc=3)
            self.gptscore.add_criterion(task=task, code=code, desc=3)
            self.gptscore.add_criterion(task=task, code=code, desc="hello")

        # Check that the attributes are correctly changed after method is applied
        self.gptscore.add_criterion(task, code, desc)

        self.assertTrue(
            self.gptscore.criteria[task] is not None
        )  # Check that the task is in the criteria
        self.assertTrue(
            code in self.gptscore.criteria[task].keys()
        )  # Check that the code is in the criteria
        self.assertTrue(
            desc == self.gptscore.criteria[task][code]
        )  # Check that the description is in the criteria

        self.assertEqual(
            self.gptscore.criteria[task][code], desc
        )  # Check that the description is in the criteria

        self.assertTrue(
            task in self.gptscore.tasks
        )  # Check that the task is in the tasks

        self.assertTrue(
            code in self.gptscore.aspects
        )  # Check that the code is in the aspects


@unittest.skip("Skip TestRelevance")
class TestRelevance(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.relevance = Relevance(llm=MODEL)

    def test_bad_arguments(self):
        query = "Which family do cats belong to?"
        prediction = "Cats are animals that are apart of the Felidae family."

        with self.assertRaises(AssertionError):
            self.relevance.compute([query], prediction)
            self.relevance.compute(query, [prediction])
            self.relevance.compute(1, prediction)
            self.relevance.compute(query, 1)
            self.relevance.compute(query, [1, 2])
            self.relevance.compute([1, 2], prediction)

    def test_relevant(self):

        query = ["Which family do cats belong to?"]
        prediction = ["Cats are animals that are apart of the Felidae family."]

        # Act
        result = self.relevance.compute(query, prediction)

        # Assert
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["value"], "Y")

    def test_irrelevant(self):

        query = ["Do I need a driving license to drive a car?"]
        prediction = [
            "Cats are animals that are apart of the Felidae family. They are often kept as pets and are carnivorous."
        ]

        # Act
        result = self.relevance.compute(query, prediction)

        # Assert
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["value"], "N")


@unittest.skip("Skip TestCorrectness")
class TestCorrectness(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.correctness = Correctness(llm=MODEL)

    def test_bad_arguments(self):
        query = "I want to know more about cats."
        prediction = "Cats are animals of the Felidae family."
        references = [
            "Cats are animals of the Felidae family.",
            "Cats are often kept as pets.",
            "Cats are carnivorous mammals.",
        ]

        with self.assertRaises(AssertionError):
            self.correctness.compute([query], prediction, references)
            self.correctness.compute(query, [prediction], references)
            self.correctness.compute(query, prediction, [references])
            self.correctness.compute(query, prediction, 1)
            self.correctness.compute(query, 1, references)
            self.correctness.compute(1, prediction, references)
            self.correctness.compute(query, prediction, [1])
            self.correctness.compute(query, [1], references)
            self.correctness.compute([1], prediction, references)

    def test_correct(self):

        query = ["I want to know more about cats."]
        references = [
            "Cats are animals of the Felidae family.",
            "Cats are often kept as pets.",
            "Cats are carnivorous mammals.",
        ]

        pred = [
            "Of course! Cats are animals of the Felidae family that are often kept as pets and are carnivorous."
        ]
        # Act
        result = self.correctness.compute(query, pred, references)
        # Assert
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["value"], "Y")

    def test_partially_correct(self):
        query = ["I want to know more about cats."]
        references = [
            "Cats are animals of the Felidae family.",
            "Cats are often kept as pets.",
            "Cats are carnivorous mammals.",
        ]

        pred = [
            "Of course! Cats are animals of the Gasterosteidae family that are often kept as pets and are carnivorous."
        ]
        # Act
        result = self.correctness.compute(query, pred, references)
        # Assert
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["value"], "N")

    def test_incorrect(self):
        query = ["I want to know more about cats."]
        references = [
            "Cats are animals of the Felidae family.",
            "Cats are often kept as pets.",
            "Cats are carnivorous mammals.",
        ]

        pred = ["If you want to drive a car, you need a driving license."]
        # Act
        result = self.correctness.compute(query, pred, references)

        # Assert
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["value"], "N")


@unittest.skip("Skip TestFaithfulness")
class TestFaithfulness(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.faithfulness = Faithfulness(llm=MODEL)

    def test_bad_arguments(self):
        user_prompts = ["Tell me about the Felidae family."]
        references = [
            "Cats are animals of the Felidae family.",
            "Cats are often kept as pets.",
        ]
        pred = "Cats are animals of the Felidae family."

        with self.assertRaises(AssertionError):
            self.faithfulness.compute([user_prompts], [pred], references)
            self.faithfulness.compute(user_prompts, [pred], [references])
            self.faithfulness.compute(user_prompts, pred, references)
            self.faithfulness.compute(user_prompts, pred, [1])
            self.faithfulness.compute(user_prompts, [1], references)
            self.faithfulness.compute([1], pred, references)

    def test_faithful(self):

        user_prompts = ["Tell me about the Felidae family."]
        references = [
            "Cats are animals of the Felidae family.",
            "Cats are often kept as pets.",
        ]
        pred = "Cats are animals of the Felidae family and are often kept as pets."

        # Act
        result = self.faithfulness.compute(user_prompts, [pred], references)

        # Assert
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["value"], "Y")

    def test_with_hallucination(self):

        user_prompts = ["Tell me about the Felidae family."]
        references = ["Cats are animals of the Felidae family."]
        pred = (
            "Cats are animals of the Felidae family, and they have wings and can fly."
        )

        # Act
        result = self.faithfulness.compute(user_prompts, [pred], references)

        # Assert
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["value"], "N")

    def test_unrelated_content(self):

        user_prompts = ["Tell me about the Felidae family."]
        references = ["Cats are animals of the Felidae family."]
        pred = "Driving a car requires a driving license."

        # Act
        result = self.faithfulness.compute(user_prompts, [pred], references)

        # Assert
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["value"], "N")

    def test_partial_faithfulness(self):

        user_prompts = ["Tell me about the Felidae family."]
        references = [
            "Cats are animals of the Felidae family.",
            "Cats are often kept as pets.",
        ]
        pred = "Cats are part of the Felidae family but they are not kept as pets."

        # Act
        result = self.faithfulness.compute(user_prompts, [pred], references)

        # Assert
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["value"], "N")

    def test_faithful_but_paraphrased(self):

        user_prompts = ["Tell me about the Felidae family."]
        references = [
            "Cats are animals of the Felidae family.",
            "Cats are often kept as pets.",
        ]
        pred = "Cats belong to the Felidae family and are usually kept as pets."

        # Act
        result = self.faithfulness.compute(user_prompts, [pred], references)

        # Assert
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["value"], "Y")


@unittest.skip("Skip TestNegativeRejection")
class TestNegativeRejection(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.negative_rejection = NegativeRejection(llm=MODEL)

    def test_bad_arguments(self):
        user_prompts = ["Tell me about cats."]
        references = ["Cats are animals of the Felidae family."]
        pred = "I'm sorry, I cannot provide information on that topic."

        with self.assertRaises(AssertionError):
            self.negative_rejection.compute([user_prompts], [pred], references)
            self.negative_rejection.compute(user_prompts, [pred], [references])
            self.negative_rejection.compute(user_prompts, pred, references)
            self.negative_rejection.compute(user_prompts, pred, [1])
            self.negative_rejection.compute(user_prompts, [1], references)
            self.negative_rejection.compute([1], pred, references)

    def test_rejection_no_evidence(self):

        user_prompts = ["Tell me about unicorns."]
        references = ["Centaurs are mythical creatures."]  # Nothing about unicorns
        pred = "I'm sorry, I cannot provide information on that topic."

        # Act
        result = self.negative_rejection.compute(user_prompts, [pred], references)

        # Assert
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["value"], "Y")

    def test_no_rejection_evidence(self):

        user_prompts = ["Tell me about cats."]
        references = ["Cats are animals of the Felidae family."]  # Evidence is present
        pred = "I'm sorry, I cannot provide information on that topic."

        # Act
        result = self.negative_rejection.compute(user_prompts, [pred], references)

        # Assert
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["value"], "N")

    def test_rejection_with_nothing(self):

        user_prompts = ["How can I apply to engineering in ULiège?"]
        references = [""]
        pred = "I'm sorry, I cannot provide information on that topic."

        # Act
        result = self.negative_rejection.compute(user_prompts, [pred], references)

        # Assert
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["value"], "Y")

    def test_no_rejection_correct_info(self):

        user_prompts = ["Tell me about dogs."]
        references = ["Dogs are domesticated mammals, not natural wild animals."]
        pred = "I'm sorry, I cannot provide information on that topic."

        # Act
        result = self.negative_rejection.compute(user_prompts, [pred], references)

        # Assert
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["value"], "N")


class TestHallucinationScore(unittest.TestCase):
    def setUp(self):
        self.hallucination_detector = HallucinationScore()

    def test_bad_arguments(self):
        predictions = ["Cats are animals of the Felidae family."]
        references = ["Cats are animals of the Felidae family."]

        with self.assertRaises(AssertionError):
            self.hallucination_detector.compute([predictions], references)
            self.hallucination_detector.compute(predictions, [references])
            self.hallucination_detector.compute(predictions, [1, 2, 3])
            self.hallucination_detector.compute([1, 2, 3], references)
            self.hallucination_detector.compute(predictions, [1])
            self.hallucination_detector.compute([1], references)

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


class TestGEvalWithMockedModel(unittest.TestCase):
    def setUp(self):
        patcher = patch("saga_llm_evaluation.helpers.utils.get_langchain_gpt_model")
        mock_get_model = patcher.start()

        self.mock_model = MagicMock(spec=BaseChatModel)
        mock_get_model.return_value = self.mock_model

        mock_response = MagicMock()
        mock_response.response_metadata = {
            "logprobs": {
                "content": [
                    {"token": "1", "logprob": -0.5},
                    {"token": ".", "logprob": -0.1},
                    {"token": "good", "logprob": -0.3},
                ]
            }
        }
        self.mock_model.invoke.return_value = mock_response
        self.geval = GEval(model=self.mock_model)

        self.addCleanup(patcher.stop)

    def test_compute_with_mock_model(self):
        source = "Hello, how are you?"
        pred = "I'm good, thanks!"
        task = "diag"
        aspect = "ENG"

        result = self.geval.compute(source, pred, task, aspect)
        self.assertTrue(result["ENG"][0] > 0)
        self.assertAlmostEqual(result["ENG"][0], 0.6065306597126334)


class TestSelfCheckGPTWithMockedModel(unittest.TestCase):
    def setUp(self):
        patcher = patch("saga_llm_evaluation.helpers.utils.get_langchain_gpt_model")
        mock_get_model = patcher.start()

        self.mock_model = MagicMock(spec=BaseChatModel)
        mock_get_model.return_value = self.mock_model

        mock_response = MagicMock()
        mock_response.content = "Yes"
        self.mock_model.invoke.return_value = mock_response

        self.selfcheckgpt = SelfCheckGPT(
            model=self.mock_model, eval_model=self.mock_model
        )

        self.addCleanup(patcher.stop)

    def test_compute_with_mock_model(self):
        user_prompt = "What is the capital of France?"
        prediction = "Paris"
        n_samples = 3

        result = self.selfcheckgpt.compute(user_prompt, prediction, n_samples)
        self.assertEqual(result[0], 1)


class TestGPTScoreWithMockedModel(unittest.TestCase):
    def setUp(self):
        patcher = patch("saga_llm_evaluation.helpers.utils.get_langchain_gpt_model")
        mock_get_model = patcher.start()

        self.mock_model = MagicMock(spec=BaseChatModel)
        mock_get_model.return_value = self.mock_model
        self.mock_model.bind.return_value = self.mock_model

        mock_response = MagicMock()
        mock_response.response_metadata = {
            "logprobs": {
                "content": [
                    {"token": "I", "logprob": -0.5},
                    {"token": "love", "logprob": -0.1},
                    {"token": "cats", "logprob": -0.3},
                    {"token": ".", "logprob": -0.1},
                ]
            }
        }
        self.mock_model.invoke.return_value = mock_response
        self.gptscore = GPTScore(model=self.mock_model)

        self.addCleanup(patcher.stop)

    def test_compute_with_mock_model(self):
        source = "How are you?"
        pred = "I'm doing great!"
        aspect = "LIK"
        task = "diag"

        result = self.gptscore.compute(source, pred, aspect=aspect, task=task)
        self.assertAlmostEqual(result["LIK"][0], -0.25)


class TestRelevanceWithMockedModel(unittest.TestCase):
    def setUp(self):
        patcher = patch("langchain.evaluation.load_evaluator")
        mock_load_evaluator = patcher.start()

        self.mock_evaluator = MagicMock()
        mock_load_evaluator.return_value = self.mock_evaluator
        self.mock_evaluator.evaluate_strings.return_value = {
            "score": 1,
            "value": "Y",
            "reasoning": "OK.",
        }

        self.relevance = Relevance()
        self.relevance.evaluator = self.mock_evaluator

        self.addCleanup(patcher.stop)

    def test_compute_with_mock_model(self):
        user_prompts = ["What is the capital of France?"]
        predictions = ["Paris is the capital of France."]

        result = self.relevance.compute(user_prompts, predictions)
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["value"], "Y")
        self.assertEqual(result["reasoning"], "OK.")


class TestCorrectnessWithMockedModel(unittest.TestCase):
    def setUp(self):
        patcher = patch("langchain.evaluation.load_evaluator")
        mock_load_evaluator = patcher.start()

        self.mock_evaluator = MagicMock()
        mock_load_evaluator.return_value = self.mock_evaluator
        self.mock_evaluator.evaluate_strings.return_value = {
            "score": 1,
            "value": "Y",
            "reasoning": "OK.",
        }

        self.correctness = Correctness()
        self.correctness.evaluator = self.mock_evaluator

        self.addCleanup(patcher.stop)

    def test_compute_with_mock_model(self):
        user_prompts = ["What is the capital of France?"]
        predictions = ["Paris is the capital of France."]
        references = ["Paris is the capital of France."]

        result = self.correctness.compute(user_prompts, predictions, references)
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["value"], "Y")
        self.assertEqual(result["reasoning"], "OK.")


class TestFaithfulnessWithMockedModel(unittest.TestCase):
    def setUp(self):
        patcher = patch("langchain.evaluation.load_evaluator")
        mock_load_evaluator = patcher.start()

        self.mock_evaluator = MagicMock()
        mock_load_evaluator.return_value = self.mock_evaluator
        self.mock_evaluator.evaluate_strings.return_value = {
            "score": 1,
            "value": "Y",
            "reasoning": "OK.",
        }

        self.faithfulness = Faithfulness()
        self.faithfulness.evaluator = self.mock_evaluator

        self.addCleanup(patcher.stop)

    def test_compute_with_mock_model(self):
        user_prompts = ["What is the capital of France?"]
        predictions = ["Paris is the capital of France."]
        references = ["Paris is the capital of France."]

        result = self.faithfulness.compute(user_prompts, predictions, references)
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["value"], "Y")
        self.assertEqual(result["reasoning"], "OK.")


class TestNegativeRejectionWithMockedModel(unittest.TestCase):
    def setUp(self):
        patcher = patch("langchain.evaluation.load_evaluator")
        mock_load_evaluator = patcher.start()

        self.mock_evaluator = MagicMock()
        mock_load_evaluator.return_value = self.mock_evaluator
        self.mock_evaluator.evaluate_strings.return_value = {
            "score": 1,
            "value": "Y",
            "reasoning": "OK.",
        }

        self.negative_rejection = NegativeRejection()
        self.negative_rejection.evaluator = self.mock_evaluator

        self.addCleanup(patcher.stop)

    def test_compute_with_mock_model(self):
        user_prompts = ["Tell me about unicorns."]
        predictions = ["I cannot provide information on that topic."]
        references = ["There is no information on unicorns."]

        result = self.negative_rejection.compute(user_prompts, predictions, references)
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["value"], "Y")
        self.assertEqual(result["reasoning"], "OK.")
