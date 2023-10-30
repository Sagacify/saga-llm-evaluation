import unittest

from saga_llm_evaluation_ml.score import LLMScorer


class TestLLMScorer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = "TheBloke/Llama-2-7b-Chat-GGUF"
        self.scorer = LLMScorer(model=self.model)

    def test_init(self):
        false = False
        with self.assertRaises(AssertionError):

            LLMScorer(model=self.model, lan=false)
            LLMScorer(model=self.model, bleurt_model=false)
            LLMScorer(model=self.model, mauve_model=false)
            LLMScorer(model=self.model, selfcheckgpt_eval_model_name_or_path=false)
            LLMScorer(model=self.model, selfcheckgpt_eval_model_basename=false)
            LLMScorer(model=self.model, geval_model_name_or_path=false)
            LLMScorer(model=self.model, geval_model_basename=false)
            LLMScorer(model=self.model, gptscore_model_name_or_path=false)
            LLMScorer(model=self.model, gptscore_model_basename=false)

    def test_score_bad_arguments(self):
        llm_input = "I am a dog."
        prompt = f"System: You are a cat. You don't like dogs. User: {llm_input}"
        context = "System: You are a cat. You don't like dogs."
        prediction = "I am a cat, I don't like dogs."
        reference = "I am a cat, I don't like dogs, miau."

        with self.assertRaises(AssertionError):
            self.scorer.score(False, prompt, context, prediction, reference)
            self.scorer.score(llm_input, False, context, prediction, reference)
            self.scorer.score(llm_input, prompt, False, prediction, reference)
            self.scorer.score(llm_input, prompt, context, False, reference)
            self.scorer.score(llm_input, prompt, context, prediction, False)
            self.scorer.score(
                llm_input, prompt, context, prediction, reference, n_samples=False
            )
            self.scorer.score(
                llm_input, prompt, context, prediction, reference, task=False
            )
            self.scorer.score(
                llm_input, prompt, context, prediction, reference, aspects=False
            )
            self.scorer.score(
                llm_input, prompt, context, prediction, reference, custom_prompt=False
            )
            self.scorer.score(
                llm_input, prompt, context, prediction, reference, custom_prompt=False
            )

    def test_score(self):
        model_name_or_path = "TheBloke/Llama-2-7b-Chat-GGUF"
        model_basename = "llama-2-7b-chat.Q2_K.gguf"  # the model is in bin format

        scorer = LLMScorer(
            model=self.model,
            selfcheckgpt_eval_model_name_or_path=model_name_or_path,
            selfcheckgpt_eval_model_basename=model_basename,
            geval_model_name_or_path=model_name_or_path,
            geval_model_basename=model_basename,
            gptscore_model_name_or_path=model_name_or_path,
            gptscore_model_basename=model_basename,
        )

        llm_input = "I am a dog."
        prompt = f"System: You are a cat. You don't like dogs. User: {llm_input}"
        context = "Examples: Eww, I hate dogs."
        prediction = "I am a cat, I don't like dogs."
        reference = "I am a cat, I don't like dogs, miau."
        task = "diag"
        aspect = ["CON"]
        custom_prompt = {
            "name": "Fluency",
            "task": "Dialog",
            "aspect": "Evaluate the fluency of the following dialog.",
        }

        scores = scorer.score(llm_input, prompt, context, prediction, reference)
        self.assertTrue(isinstance(scores, dict))
        self.assertTrue("metrics" in scores)
        self.assertTrue("metadata" in scores)

        # All default
        print("All default")
        scores = scorer.score(llm_input, prompt, prediction, n_samples=2)
        self.assertTrue(isinstance(scores, dict))
        self.assertTrue("metrics" in scores)
        self.assertTrue("metadata" in scores)

        # All default, but with context
        print("All default, but with context")
        scores = scorer.score(
            llm_input,
            prompt,
            prediction,
            context=context,
            n_samples=2,
        )
        self.assertTrue(isinstance(scores, dict))
        self.assertTrue("metrics" in scores)
        self.assertTrue("metadata" in scores)

        # All default, but with reference
        print("All default, but with reference")
        scores = scorer.score(
            llm_input,
            prompt,
            prediction,
            reference=reference,
            n_samples=2,
        )
        self.assertTrue(isinstance(scores, dict))
        self.assertTrue("metrics" in scores)
        self.assertTrue("metadata" in scores)

        # Precise task and aspect
        print("Precise task and aspect")
        scores = scorer.score(
            llm_input,
            prompt,
            prediction,
            task=task,
            aspects=aspect,
            n_samples=2,
        )
        self.assertTrue(isinstance(scores, dict))
        self.assertTrue("metrics" in scores)
        self.assertTrue("metadata" in scores)

        # Precise custom prompt
        print("Precise custom prompt")
        scores = scorer.score(
            llm_input,
            prompt,
            prediction,
            custom_prompt=custom_prompt,
            n_samples=2,
        )
        self.assertTrue(isinstance(scores, dict))
        self.assertTrue("metrics" in scores)
        self.assertTrue("metadata" in scores)
