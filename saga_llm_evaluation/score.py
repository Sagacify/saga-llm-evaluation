from saga_llm_evaluation.helpers.embedding_metrics import MAUVE, BERTScore
from saga_llm_evaluation.helpers.language_metrics import BLEURTScore, QSquared
from saga_llm_evaluation.helpers.llm_metrics import GEval, GPTScore, SelfCheckGPT
from saga_llm_evaluation.helpers.utils import (
    MetadataExtractor,
    check_list_type,
    filter_class_input,
    load_json,
)

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=dangerous-default-value
# pylint: disable=unidiomatic-typecheck
# pylint: disable=too-many-return-statements


def get_model(config: dict, model=None, eval_model=None, key: str = "bert_score"):
    """
    Get the evaluation metric model.

    Args:
        config (dict): Config file.
        model (object, optional): Model to evaluate. Defaults to None.
        eval_model (object, optional): Evaluation model. Defaults to None.
        key (str, optional): Evaluation metric to use. Defaults to "bert_score".

    Returns:
        object: Evaluation metric model.
    """

    if key == "bert_score":
        # filter out keys that are not supported by BERTScore class
        args = filter_class_input(config[key], BERTScore.__init__)
        return BERTScore(**args)
    if key == "mauve":
        args = filter_class_input(config[key], MAUVE.__init__)
        return MAUVE(**args)
    if key == "bleurt":
        args = filter_class_input(config[key], BLEURTScore.__init__)
        return BLEURTScore(**args)
    if key == "q_squared":
        args = filter_class_input(config[key], QSquared.__init__)
        return QSquared(**args)
    if key == "selfcheckgpt":
        assert (
            model is not None
        ), "model to evaluate must be defined as an input parameter for SelfCheckGPT."
        args = filter_class_input(config[key], SelfCheckGPT.__init__)
        return SelfCheckGPT(model=model, eval_model=eval_model, **args)
    if key == "geval":
        args = filter_class_input(config[key], GEval.__init__)
        return GEval(model=eval_model, **args)
    if key == "gptscore":
        args = filter_class_input(config[key], GPTScore.__init__)
        return GPTScore(model=eval_model, **args)
    raise ValueError(
        f"Model {key} not found. It must be one of the following: bert_score, mauve, bleurt, \
            q_squared, selfcheckgpt, geval, gptscore"
    )


class LLMScorer:
    def __init__(
        self,
        metrics=[
            "bert_score",
            "mauve",
            "bleurt",
            "q_squared",
            "selfcheckgpt",
            "geval",
            "gptscore",
        ],
        model=None,
        eval_model=None,
        config=None,
    ) -> None:
        """
        Initialize the LLMScorer class. This class is used to evaluate the performance of a language model\
        using a set of evaluation metrics.\
        The evaluation metrics are defined in the config file or can be passed as an input parameter.\
        The model to evaluate and the evaluation model can be passed as input parameters.

        Args:
            metrics (list, optional): List of evaluation metrics to use.\
                Defaults to ["bert_score", "mauve", "bleurt", "q_squared", "selfcheckgpt", "geval", "gptscore"].
            model (object, optional): Model to evaluate. Defaults to None.
            eval_model (object, optional): Evaluation model. Defaults to None.
            config (dict, optional): Config file. Defaults to None.
        """

        self.config = (
            config if config else load_json("./saga_llm_evaluation/scorer.json")
        )
        assert isinstance(metrics, list), "metrics must be a list."
        assert isinstance(self.config, dict), "config file must be a dict."
        assert all(
            metric in self.config["metrics"] for metric in metrics
        ), "config file must have a configuration for each evaluation metric defined."

        # Metrics
        self.metrics = {}

        for metric in metrics:
            self.metrics[metric] = get_model(
                self.config["metrics"], model, eval_model, metric
            )
        # Metadata
        self.metadata_extractor = MetadataExtractor()

    def add_geval_task(self, name: str, definition: str):
        """
        This function adds a new task to the GEval metric.
        Please follow the example pattern below to ensure consistency.

        Example:

        .. code-block:: python

            "summ": "You will be given one summary written for a news article.\\n"
                    "Your task is to rate the summary on one metric.\\n"
                    "Please make sure you read and understand these instructions carefully.\\n"
                    "Please keep this document open while reviewing, and refer to it as needed."

        Args:
            name (str): Name of the task.
            definition (str): Definition of the task.
        """
        assert isinstance(name, str), "name must be a string."
        assert isinstance(definition, str), "definition must be a string."

        if "geval" in self.metrics:
            self.metrics["geval"].add_task(name, definition)
        else:
            raise TypeError("GEval metric is not defined.")

    def add_geval_aspect(self, code: str, name: str, prompt: str):
        """
        This function adds a new aspect to the GEval metric.
        Please follow the example pattern below to ensure consistency.

        Example:

        .. code-block:: python

            "COH": {
                "name": "Coherence",
                "prompt": "Coherence (1-5) - the overall quality and logical flow of all sentences.\\
                    This dimension aligns with the DUC quality question of structure and coherence, which states that\\
                    the summary should be well-structured and well-organized. It should not just be\\
                    a collection of related information, but should build from sentence to sentence\\
                    to form a coherent body of information about a topic."
            }

        Args:
            code (str): Code of the aspect.
            name (str): Name of the aspect.
            prompt (str): Prompt of the aspect.
        """
        assert isinstance(code, str), "code must be a string."
        assert isinstance(name, str), "name must be a string."
        assert isinstance(prompt, str), "prompt must be a string."

        if "geval" in self.metrics:
            self.metrics["geval"].add_aspect(code, name, prompt)
        else:
            raise TypeError("GEval metric is not defined.")

    def add_gptscore_template(self, task: str, code: str, prompt: str):
        """
        This function adds a template to the GPTScore metric.
        Please follow the example pattern below to ensure consistency.

        Example:

        .. code-block:: python

            "diag": {
                "COH": (
                    f"Answer the question based on the conversation between a human and AI.\\n"
                    "Question: Is the AI coherent and maintains a good conversation flow throughout the conversation? (a) Yes. (b) No.\\n"
                    "Conversation:\\nUser: {{src}}\\nAI: {{pred}}\\nAnswer: Yes."
                ),
            }

        Args:
            task (str): Task of the template.
            code (str): Code of the aspect.
            prompt (str): Prompt of the aspect.
        """
        assert isinstance(task, str), "task must be a string."
        assert isinstance(code, str), "code must be a string."
        assert isinstance(prompt, str), "prompt must be a string."

        if "gptscore" in self.metrics:
            self.metrics["gptscore"].add_template(task, code, prompt)
        else:
            raise TypeError("GPTScore metric is not defined.")

    def score(
        self,
        user_prompt: list,
        prediction: list,
        knowledge: list = None,
        reference: list = None,
        config: dict = None,
    ):
        """
        This function computes the evaluation metrics for a given user prompt and prediction.

        Args:
            user_prompt (str): user prompt to the model
            prediction (str): Prediction of the model.
            knowledge (str, optional): Source text that the model used to generate the prediction. Defaults to None.
            reference (str, optional): Reference of the prediction. Defaults to None.
            config (dict, optional): Config file. Defaults to None.

        Returns:
            dict: Dictionary containing the metadata and evaluation metrics.
        """
        # TODO: add support to custom_prompt for geval and gptscore thruogh the config file
        # Custom prompt template. Defaults to None.
        #        Must contain the following keys: "task", "aspect", "name".

        self.config = config if config else self.config
        assert isinstance(self.config, dict), "config file must be a dict."
        required_inputs = [user_prompt, prediction]
        assert all(isinstance(inpt, str) for inpt in required_inputs) or (
            all(check_list_type(inpt, str) for inpt in required_inputs)
            and len(set(map(len, required_inputs))) == 1
        ), "user_prompt and prediction must be strings or lists of strings with equal size."
        assert type(user_prompt) == type(
            prediction
        ), "user_prompt and prediction must be of the same type."

        # check types
        assert (
            isinstance(knowledge, str)
            or check_list_type(knowledge, str)
            or knowledge is None
        ), "If knowledge (context) is passed, it must be either a list or a string."
        assert type(user_prompt) == type(
            knowledge
        ), "user_prompt, prediction and knowledge must be of the same type."
        assert (
            isinstance(reference, str)
            or check_list_type(reference, str)
            or reference is None
        ), "If reference is passed , it must be a string or None."
        assert type(user_prompt) == type(
            reference
        ), "user_prompt, prediction and reference must be of the same type."

        # if str is passed, convert to list
        if isinstance(user_prompt, str):  # all input variables shuold be strings
            user_prompt = [user_prompt]
            prediction = [prediction]
            knowledge = [knowledge] if knowledge else None
            reference = [reference] if reference else None

        if knowledge is not None:
            assert (
                len(set(map(len, [user_prompt, knowledge]))) == 1
            ), "user_prompt and knowledge must have the same size."
        if reference is not None:
            assert (
                len(set(map(len, [user_prompt, reference]))) == 1
            ), "user_prompt and reference must have the same size."

        metadata_dict = {
            "user_prompt": [
                self.metadata_extractor.compute(prompt) for prompt in user_prompt
            ],
            "prediction": [
                self.metadata_extractor.compute(pred) for pred in prediction
            ],
            "knowledge": [self.metadata_extractor.compute(know) for know in knowledge]
            if knowledge
            else None,
            "reference": [self.metadata_extractor.compute(ref) for ref in reference]
            if reference
            else None,
        }

        input_config = {
            "user_prompts": user_prompt,
            "predictions": prediction,
            "knowledges": knowledge,
            "references": reference,
        }

        evaluation = {}
        for metric, metric_config in self.config["metrics"].items():

            # keep args needed by the compute function of the evaluation metric
            input_args = filter_class_input(
                args=input_config, python_function=self.metrics[metric].compute
            )
            compute_args = filter_class_input(
                args=metric_config, python_function=self.metrics[metric].compute
            )
            input_args.update(compute_args)
            evaluation[metric] = self.metrics[metric].compute(**input_args)

        return {"metadata": metadata_dict, "evaluation": evaluation}
