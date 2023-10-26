from saga_llm_evaluation_ml.helpers.embedding_metrics import MAUVE, BERTScore
from saga_llm_evaluation_ml.helpers.language_metrics import BLEURTScore, QSquared
from saga_llm_evaluation_ml.helpers.llm_metrics import SelfCheckGPT, GEval, GPTScore
from saga_llm_evaluation_ml.helpers.utils import MetadataExtractor


class LLMScorer:
    def __init__(
        self,
        model,
        lan="en",
        bleurt_model="BLEURT-tiny",
        mauve_model="gpt2",
        eval_model_name_or_path="TheBloke/Llama-2-7b-Chat-GGUF",
        eval_model_basename="llama-2-7b-chat.Q4_K_M.gguf",
        model_name_or_path="TheBloke/Llama-2-7b-Chat-GGUF",
        model_basename="llama-2-7b-chat.Q4_K_M.gguf",
    ) -> None:
        assert isinstance(lan, str), "lan must be a string."
        assert isinstance(bleurt_model, str), "bleurt_model must be a string."
        assert isinstance(mauve_model, str), "mauve_model must be a string."
        assert isinstance(
            eval_model_name_or_path, str
        ), "eval_model_name_or_path must be a string."
        assert isinstance(
            eval_model_basename, str
        ), "eval_model_basename must be a string."
        assert isinstance(
            model_name_or_path, str
        ), "model_name_or_path must be a string."
        assert isinstance(model_basename, str), "model_basename must be a string."

        # Metrics
        self.bert_score = BERTScore(lan=lan)
        self.mauve = MAUVE(featurize_model_name=mauve_model)
        self.bleurt_score = BLEURTScore(checkpoint=bleurt_model)
        self.q_squared = QSquared(lan=lan)
        self.selfcheckgpt = SelfCheckGPT(
            model,
            eval_model_name_or_path=eval_model_name_or_path,
            eval_model_basename=eval_model_basename,
        )
        self.geval = GEval(
            model_name_or_path=model_name_or_path, model_basename=model_basename
        )
        self.gptscore = GPTScore(
            model_name_or_path=model_name_or_path, model_basename=model_basename
        )

        # Metadata
        self.metadata_extractor = MetadataExtractor()

    def score(
        self,
        llm_input: str,
        prompt: str,
        prediction: str,
        context: str = None,
        reference: str = None,
        n_samples: int = 5,
        task: str = None,
        aspects: list = None,
        custom_prompt: dict = None,
    ):
        """
        Args:
            llm_input (str): llm_input to the model.
            prompt (str): Prompt to the model. Comprises the context and the llm_input.
            prediction (str): Prediction of the model.
            context (str, optional): Context of the prediction. Defaults to None.
            reference (str, optional): Reference of the prediction. Defaults to None.
            n_samples (int, optional): Number of samples to generate. Defaults to 5.
            task (str, optional): Task definition. Defaults to None.
            aspects (list, optional): Aspects to evaluate. Defaults to None.
            custom_prompt (dict, optional): Custom prompt template. Defaults to None.
                Must contain the following keys: "task", "aspect", "name".
        """
        assert isinstance(prompt, str), "prompt must be a string."
        assert isinstance(llm_input, str), "llm_input must be a string."
        assert isinstance(prediction, str), "prediction must be a string."
        assert isinstance(context, str) or context is None, "context must be a string."
        assert (
            isinstance(reference, str) or reference is None
        ), "Reference must be a string or None."
        assert isinstance(n_samples, int), "n_samples must be an integer."
        assert n_samples > 0, "n_samples must be greater than 0."
        assert isinstance(task, str) or task is None, "task must be a string or None."
        assert (
            isinstance(aspects, list) or aspects is None
        ), "aspects must be a list or None."
        assert (
            isinstance(custom_prompt, dict) or custom_prompt is None
        ), "custom_prompt must be a dict or None."
        if isinstance(custom_prompt, dict):
            assert (
                "task" in custom_prompt.keys()
                and "aspect" in custom_prompt.keys()
                and "name" in custom_prompt.keys()
            ), "custom_prompt must contain the following keys: 'task', 'aspect', 'name'."

        if aspects:
            geval_scores = {}
            gpt_scores = {}
            for aspect in aspects:
                geval_scores[aspect] = self.geval.compute(
                    prompt, prediction, task, aspect, custom_prompt
                )
                gpt_scores[aspect] = self.gptscore.compute(
                    prompt, prediction, custom_prompt, aspect, task
                )

        metadata_dict = {
            "prompt": self.metadata_extractor.compute(prompt),
            "llm_input": self.metadata_extractor.compute(llm_input),
            "prediction": self.metadata_extractor.compute(prediction),
            "context": self.metadata_extractor.compute(context) if context else None,
            "reference": self.metadata_extractor.compute(reference)
            if reference
            else None,
        }

        metrics_dict = {
            "bert_score": self.bert_score.compute([reference], [prediction])
            if reference
            else None,
            "mauve": self.mauve.compute([reference], [prediction])
            if reference
            else None,
            "bleurt_score": self.bleurt_score.compute([reference], [prediction])
            if reference
            else None,
            "q_squared": self.q_squared.compute(prediction, context),
            "selfcheck_gpt": self.selfcheckgpt.compute(prompt, prediction, n_samples),
            "g_eval": self.geval.compute(
                prompt, prediction, custom_prompt=custom_prompt
            )
            if custom_prompt
            else geval_scores
            if aspects and task
            else None,
            "gpt_score": self.gptscore.compute(prompt, prediction, prompt=custom_prompt)
            if custom_prompt
            else gpt_scores
            if aspects and task
            else None,
        }

        return {"metadata": metadata_dict, "metrics": metrics_dict}
