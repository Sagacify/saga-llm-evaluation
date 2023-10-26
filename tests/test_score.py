import unittest
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from saga_llm_evaluation_ml.score import LLMScorer


class TestLLMScorer(unittest.TestCase):
    def test_init(self):
        model = "TheBloke/Llama-2-7b-Chat-GGUF"
        false = False
        with self.assertRaises(AssertionError):

            LLMScorer(model=model, lan=false)
            LLMScorer(model=model, bleurt_model=false)
            LLMScorer(model=model, mauve_model=false)
            LLMScorer(model=model, eval_model_name_or_path=false)
            LLMScorer(model=model, eval_model_basename=false)
            LLMScorer(model=model, model_name_or_path=false)
            LLMScorer(model=model, model_basename=false)

    def test_score_bad_arguments(self):
        model = "TheBloke/Llama-2-7b-Chat-GGUF"
        scorer = LLMScorer(model=model)

        llm_input = "I am a dog."
        prompt = f"System: You are a cat. You don't like dogs. User: {llm_input}"
        context = "System: You are a cat. You don't like dogs."
        prediction = "I am a cat, I don't like dogs."
        reference = "I am a cat, I don't like dogs, miau."

        with self.assertRaises(AssertionError):
            scorer.score(False, prompt, context, prediction, reference)
            scorer.score(llm_input, False, context, prediction, reference)
            scorer.score(llm_input, prompt, False, prediction, reference)
            scorer.score(llm_input, prompt, context, False, reference)
            scorer.score(llm_input, prompt, context, prediction, False)
            scorer.score(
                llm_input, prompt, context, prediction, reference, n_samples=False
            )
            scorer.score(llm_input, prompt, context, prediction, reference, task=False)
            scorer.score(
                llm_input, prompt, context, prediction, reference, aspects=False
            )
            scorer.score(
                llm_input, prompt, context, prediction, reference, custom_prompt=False
            )
            scorer.score(
                llm_input, prompt, context, prediction, reference, custom_prompt=False
            )

    def test_score(self):
        model_name_or_path = "TheBloke/Llama-2-7b-Chat-GGUF"
        model_basename = "llama-2-7b-chat.Q2_K.gguf"  # the model is in bin format

        model_path = hf_hub_download(
            repo_id=model_name_or_path, filename=model_basename
        )
        model = Llama(model_path=model_path, n_threads=2, verbose=False)  # CPU cores

        scorer = LLMScorer(
            model=model,
            eval_model_name_or_path=model_name_or_path,
            eval_model_basename=model_basename,
            model_name_or_path=model_name_or_path,
            model_basename=model_basename,
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
