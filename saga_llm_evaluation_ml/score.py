from saga_llm_evaluation_ml.helpers.embedding_metrics import MAUVE, BERTScore
from saga_llm_evaluation_ml.helpers.language_metrics import BLEURTScore, QSquared
from saga_llm_evaluation_ml.helpers.llm_metrics import GEval, GPTScore, SelfCheckGPT
from saga_llm_evaluation_ml.helpers.utils import MetadataExtractor


class LLMScorer:
    # pylint: disable=invalid-name
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        model=None,
        lan="en",
        bleurt_model="BLEURT-tiny",
        mauve_model="gpt2",
        selfcheckgpt_eval_model_name_or_path="TheBloke/Llama-2-7b-Chat-GGUF",
        selfcheckgpt_eval_model_basename="llama-2-7b-chat.Q4_K_M.gguf",
        geval_model_name_or_path="TheBloke/Llama-2-7b-Chat-GGUF",
        geval_model_basename="llama-2-7b-chat.Q4_K_M.gguf",
        gptscore_model_name_or_path="TheBloke/Llama-2-7b-Chat-GGUF",
        gptscore_model_basename="llama-2-7b-chat.Q4_K_M.gguf",
    ) -> None:
        assert isinstance(lan, str), "lan must be a string."
        assert isinstance(bleurt_model, str), "bleurt_model must be a string."
        assert isinstance(mauve_model, str), "mauve_model must be a string."
        assert isinstance(
            selfcheckgpt_eval_model_name_or_path, str
        ), "selfcheckgpt_eval_model_name_or_path must be a string."
        assert isinstance(
            selfcheckgpt_eval_model_basename, str
        ), "selfcheckgpt_eval_model_basename must be a string."
        assert isinstance(
            geval_model_name_or_path, str
        ), "geval_model_name_or_path must be a string."
        assert isinstance(
            geval_model_basename, str
        ), "geval_model_basename must be a string."
        assert isinstance(
            gptscore_model_name_or_path, str
        ), "gptscore_model_name_or_path must be a string."
        assert isinstance(
            gptscore_model_basename, str
        ), "gptscore_model_basename must be a string."

        # Metrics
        self.bert_score = BERTScore(lan=lan)
        self.mauve = MAUVE(featurize_model_name=mauve_model)
        self.bleurt_score = BLEURTScore(checkpoint=bleurt_model)
        self.q_squared = QSquared(lan=lan)
        self.selfcheckgpt = (
            None
            if model is None
            else SelfCheckGPT(
                model,
                eval_model_name_or_path=selfcheckgpt_eval_model_name_or_path,
                eval_model_basename=selfcheckgpt_eval_model_basename,
            )
        )
        self.geval = GEval(
            model_name_or_path=geval_model_name_or_path,
            model_basename=geval_model_basename,
        )
        self.gptscore = GPTScore(
            model_name_or_path=gptscore_model_name_or_path,
            model_basename=gptscore_model_basename,
        )

        # Metadata
        self.metadata_extractor = MetadataExtractor()

    def add_geval_task(self, name, definition):
        """
        This function adds a task to the GEval metric.
        Please try to follow the following example pattern to ensure consistency.
        Example:
        "summ": "You will be given one summary written for a news article.
            Your task is to rate the summary on one metric.
            Please make sure you read and understand these instructions carefully.
            Please keep this document open while reviewing, and refer to it as needed.",
        Args:
            name (str): Name of the task.
            definition (str): Definition of the task.
        """
        assert isinstance(name, str), "name must be a string."
        assert isinstance(definition, str), "definition must be a string."

        if self.geval is not None:
            self.geval.add_task(name, definition)
        else:
            raise TypeError("GEval metric is not defined.")

    def add_geval_aspect(self, code, name, prompt):
        """
        This function adds an aspect to the GEval metric.
        Please try to follow the following example pattern to ensure consistency.
        Example:
        "COH": {
            "name": "Coherence",
            "prompt": "Coherence (1-5) - the collective quality of all sentences.
                We align this dimension with the DUC quality question of structure and coherence
                whereby ”the summary should be well-structured and well-organized.
                The summary should not just be a heap of related information,
                but should build from sentence to sentence to a coherent body of information about a topic.”",
        },

        Args:
            code (str): Code of the aspect.
            name (str): Name of the aspect.
            prompt (str): Prompt of the aspect.
        """
        assert isinstance(code, str), "code must be a string."
        assert isinstance(name, str), "name must be a string."
        assert isinstance(prompt, str), "prompt must be a string."

        if self.geval is not None:
            self.geval.add_aspect(code, name, prompt)
        else:
            raise TypeError("GEval metric is not defined.")

    def add_gptscore_template(self, task, code, prompt):
        """
        This function adds a template to the GPTScore metric.
        Please try to follow the following example pattern to ensure consistency.
        Example:
        "diag": {
            "COH": f"Answer the question based on the conversation between a human and AI.
            \nQuestion: Is the AI coherent and maintains a good conversation flow throughout the conversation?
            (a) Yes. (b) No.\nConversation:\nUser: {{src}}\nAI: {{pred}}\nAnswer:",
        }
        Args:
            task (str): Task of the template.
            code (str): Code of the aspect.
            prompt (str): Prompt of the aspect.
        """
        assert isinstance(task, str), "task must be a string."
        assert isinstance(code, str), "code must be a string."
        assert isinstance(prompt, str), "prompt must be a string."

        if self.gptscore is not None:
            self.gptscore.add_template(task, code, prompt)
        else:
            raise TypeError("GPTScore metric is not defined.")

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
            "selfcheck_gpt": None
            if self.selfcheckgpt is None
            else self.selfcheckgpt.compute(llm_input, prediction, n_samples),
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
