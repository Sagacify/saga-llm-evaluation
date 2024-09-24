import numpy as np

from langchain.evaluation import load_evaluator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.base import BaseLanguageModel


from saga_llm_evaluation.helpers.utils import (
    check_list_type,
    get_langchain_gpt_model,
    get_langchain_llama_model,
)

# pylint: disable=consider-iterating-dictionary
# pylint: disable=too-many-locals
# pylint: disable=unidiomatic-typecheck


class SelfCheckGPT:
    def __init__(
        self,
        model,
        eval_model=None,
    ):
        """
        This class implements the self-check GPT evaluation metric for generative language models.
        It is inspired by the self-check metric proposed in https://arxiv.org/pdf/2303.08896.pdf.

        Args:
            model (Langchain BaseChatModel): LLM model to evaluate.
            eval_model (Langchain BaseChatModel, optional): Evaluation model. If None, the model used\
                is "llama" by default.
        """
        # Check that provided model is indeed BaseChatModel from LangChain
        assert model is not None and isinstance(
            model, BaseChatModel
        ), f"model must be a LangChain BaseChatModel. model is of type {type(model).__name__}"
        assert eval_model is None or isinstance(
            eval_model, BaseChatModel
        ), f"eval_model must be a LangChain BaseChatModel. eval_model is of type {type(eval_model).__name__}"

        self.model = model

        if eval_model:
            self.eval_model = eval_model
        else:  # Use Llama2 by default #TODO: do I let llama by default since this one works with it ?
            self.eval_model = get_langchain_llama_model()

    def get_prompt(self, pred: str, sample: str, question: str):
        """
        This method returns a prompt template given a candidate sentence, a sample sentence, and a question.

        Args:
            pred (str): Candidate sentence.
            sample (str): Sample sentence.
            question (str): Question asked to the model for which it generated the candidate sentence.

        Returns:
            str: Prompt template.
        """
        system_prompt = "You are a helpful, polite and concise assistant. Your task is to check if two texts provide the same answer to a given question. Always answer with a single word. The possible answers are either YES or NO.\n\n"
        question = "###Question:\n" + question
        text1 = "\n###Text 1: " + sample
        text2 = "\n###Text 2: " + pred

        message = [
            ("system", system_prompt),
            ("human", question + text1 + text2),
        ]

        return message

    def get_prompts(self, pred: str, samples: str, question: str):
        """
        This method returns a list of prompt templates given a candidate sentence, a list\
        of sample sentences, and a question.

        Args:
            pred (str): Candidate sentence.
            samples (list of str): List of sample sentences.
            question (str): Question asked to the model for which it generated the candidate sentence.

        Returns:
            list: List of prompt templates.
        """
        return [self.get_prompt(pred, sample, question) for sample in samples]

    def compute(self, user_prompts: list, predictions: list, n_samples=5):
        """
        This method computes the self-check GPT score for a candidate sentence given a source text,\
        a prompt template, and a question.

        Args:
            user_prompts (str): Question asked to the model for which it generated the candidate sentence.
            predictions (str): Candidate sentence.
            n_samples (int): Number of samples to generate.

        Returns:
            float: Score for the candidate sentence.
        """
        assert isinstance(user_prompts, str) or check_list_type(
            user_prompts, str
        ), "user_prompts must be either a list of string or a string."
        assert isinstance(predictions, str) or check_list_type(
            predictions, str
        ), "predictions must be either a list of string or a string."
        assert type(predictions) == type(
            user_prompts
        ), "user_prompts and predictions must be of the same type."
        assert isinstance(n_samples, int), "Number of samples must be an integer."
        assert n_samples > 0, "Number of samples must be greater than 0."

        # map to list if string
        if isinstance(
            user_prompts, str
        ):  # all input variables shuold be strings in this case
            user_prompts = [user_prompts]
            predictions = [predictions]

        scores = []
        for user_prompt, prediction in zip(user_prompts, predictions):
            # Generate n_samples samples from the model
            samples = []
            for _ in range(n_samples):
                system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible."
                messages = [("system", system_prompt), ("human", user_prompt)]
                response = self.model.invoke(messages)
                samples.append(response.content)

            # For each sample, ask evaluator model to evaluate the sample
            prompts = self.get_prompts(prediction, samples, user_prompt)
            sample_scores = []
            for prompt in prompts:
                answer = self.eval_model.invoke(prompt).content
                sample_scores.append(answer)

            # Compute the score: how often the sentence if supported by the sample
            scores.append(
                np.mean([1 if "yes" in score.lower() else 0 for score in sample_scores])
            )

        return scores


class GEval:
    def __init__(
        self,
        model=None,
    ):
        """
        This class implements the GEval evaluation metric for generative language models.
        It is inspired by the GEval metric proposed in https://arxiv.org/pdf/2303.16634.pdf.

        Args:
            model (LangChain BaseChatModel): model used for evaluation. If False, the model used\
                is "gpt-3.5-turbo" by default.
        """

        # Check that provided model is indeed BaseChatModel from LangChain
        assert model is None or isinstance(
            model, BaseChatModel
        ), f"model must be a LangChain BaseChatModel. model is of type {type(model).__name__}"
        if model:
            self.model = model
        else:  # Use GPT 3.5 by default if no other model provided
            self.model = get_langchain_gpt_model()

        self.tasks = {
            "summ": "You will be given one summary written for a news article. Your task is to rate the summary on one metric. Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.",
            "diag": "You will be given a conversation between two individuals. You will then be given one potential response for the next turn in the conversation. The response concerns an interesting fact, which will be provided as well. Your task is to rate the responses on one metric. Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.",
        }
        self.aspects = {
            "COH": {
                "name": "Coherence",
                "prompt": "Coherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby ”the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to sentence to a coherent body of information about a topic.”",
            },
            "CON": {
                "name": "Consistency",
                "prompt": "Consistency (1-5) - the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Annotators were also asked to penalize summaries that contained hallucinated facts. ",
            },
            "ENG": {
                "name": "Engagingness",
                "prompt": "Engagingness (1-5) - Is the response dull/interesting? - A score of 1 indicates that the response is dull and uninteresting. A score of 5 indicates that the response is interesting and engaging.",
            },
            "FLU": {
                "name": "Fluency",
                "prompt": "Fluency (1-5) - the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure. - 1: Poor. The summary is difficult to read and understand. It contains many grammatical errors, spelling mistakes, and/or punctuation errors. - 2: Fair. The summary is somewhat difficult to read and understand. It contains some grammatical errors, spelling mistakes, and/or punctuation errors. - 3: Good. The summary is easy to read and understand. It contains few grammatical errors, spelling mistakes, and/or punctuation errors. - 4: Very Good. The summary is easy to read and understand. It contains no grammatical errors, spelling mistakes, and/or punctuation errors. - 5: Excellent. The summary is easy to read and understand. It contains no grammatical errors, spelling mistakes, and/or punctuation errors.",
            },
            "REL": {
                "name": "Relevance",
                "prompt": "Relevance (1-5) - selection of important content from the source. The summary should include only important information from the source document. Annotators were instructed to penalize summaries which contained redundancies and excess information.",
            },
            "POL": {
                "name": "Politeness",
                "prompt": "Politeness (1-5) - the degree to which the response is polite. - 1: Very impolite. The response is very impolite. - 2: Somewhat impolite. The response is somewhat impolite. - 3: Neutral. The response is neutral. - 4: Somewhat polite. The response is somewhat polite. - 5: Very polite. The response is very polite.",
            },
        }

    def add_task(self, name: str, definition: str):
        """
        This method adds a task to the list of pre-defined tasks.
        Please try to follow the following example pattern to ensure consistency.
        Example:

        .. code-block:: python

            "summ": "You will be given one summary written for a news article.\\n"
                    "Your task is to rate the summary on one metric.\\n"
                    "Please make sure you read and understand these instructions carefully.\\n"
                    "Please keep this document open while reviewing, and refer to it as needed."

        Args:
            name (str): Task name.
            definition (str): Task description.
        """
        assert isinstance(name, str), "name must be a string."
        assert isinstance(definition, str), "definition must be a string."

        self.tasks[name] = definition

    def add_aspect(self, code: str, name: str, prompt: str):
        """
        This method adds an aspect to the list of pre-defined aspects.
        Please try to follow the following example pattern to ensure consistency.
        
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
            code (str): Aspect code.
            name (str): Aspect name.
            prompt (str): Aspect prompt.
        """
        assert isinstance(code, str), "code must be a string."
        assert isinstance(name, str), "name must be a string."
        assert isinstance(prompt, str), "prompt must be a string."

        self.aspects[code] = {"name": name, "prompt": prompt}

    def get_prediction(self, prompt: str):
        """
        This method returns a prediction given a prompt template.

        Args:
            prompt (str): Prompt template.

        Returns:
            dict: Response from the model.
        """
        response = self.model.invoke(prompt, top_p=0.95, logprobs=True)
        return response

    def get_cot(self, prompt: str):
        """
        This method returns a chain of thoughts given a prompt template.

        Args:
            prompt (str): Prompt template.

        Returns:
            str: Chain of thoughts.
        """
        title = "\nEvaluation steps:\n"

        message = (
            ("system", prompt + title),
            (
                "human",
                "Please provide a step-by-step chain of thoughts. Be concise: Reply with only the steps.",
            ),
        )
        cot = self.get_prediction(message).content
        return cot

    def get_prompt(
        self,
        prompts: list,
        predictions: list,
        task: str,
        aspect: str,
        custom_prompt: dict = None,
    ):
        """
        This method returns a prompt template given a source text, a candidate sentence, an aspect to evaluate,
        and a task description.

        Args:
            prompts (list): list of source text.
            predictions (list): list of candidate sentence to evaluate.
            task (str): Definition of the task.
            aspect (str): Evaluation criterion code.
            custom_prompt (dict): Custom prompt template.\
                Must contain the following keys: "task", "aspect", "name".

        Returns:
            list: List of prompt templates
        """
        definition = (
            "\n Task definition:\n" + self.tasks[task]
            if task in self.tasks.keys()
            else custom_prompt["task"]
        )
        crit = (
            "\n Evaluation criteria:\n" + self.aspects[aspect]["prompt"]
            if aspect in self.aspects.keys()
            else custom_prompt["aspect"]
        )

        # get geval_prompt with chain of thoughts, set of intermediate instructions generated
        # by llm detailing evaluation steps
        geval_prompt = f"{definition} {crit}"
        auto_cot = self.get_cot(geval_prompt)

        geval_prompts = []
        for prompt, prediction in zip(prompts, predictions):
            message = [
                ("system", geval_prompt + auto_cot),
                (
                    "human",
                    "Input Text:\n"
                    + prompt
                    + "\n Output text:\n"
                    + prediction
                    + "\n What is the score ? (answer score only)",
                ),
            ]
            geval_prompts.append(message)

        return geval_prompts

    def get_score(self, prompts: list):
        """
        This method returns the GEval score given a prompt template.

        Args:
            prompts (list): List of prompt template.

        Returns:
            list: List of scores for each candidate sentence.
        """
        scores = []
        for prompt in prompts:
            response = self.get_prediction(prompt)

            response_metadata = response.response_metadata
            tokens = [
                entry["token"] for entry in response_metadata["logprobs"]["content"]
            ]
            top_logprobs = [
                entry["logprob"] for entry in response_metadata["logprobs"]["content"]
            ]

            # Extract number index from the remaining tokens
            number_index = None
            for token in tokens:
                if token.isdigit():
                    number_index = tokens.index(token)
                    break
            if number_index is None:
                raise ValueError("Number not found in the tokens.")

            # Get logprobs associated with number in a dictionary
            logprobs = {
                tokens[i]: top_logprobs[i] for i in range(number_index, len(tokens))
            }

            # Compute score
            # Get only keys that are numbers
            number_keys = [int(key) for key in logprobs.keys() if key.isdigit()]
            number_logprobs = [logprobs[str(key)] for key in number_keys]
            number_probs = [np.exp(logprob) for logprob in number_logprobs]

            score = np.sum(np.multiply(number_keys, number_probs)) / len(number_keys)
            scores.append(score)

        return scores

    def compute(
        self,
        user_prompts: list,
        predictions: list,
        task=None,
        aspect=None,
        custom_prompt=None,
    ):
        """
        This method computes the GEval score for a candidate sentence given a source text,
        a prompt template, an aspect to evaluate, and a task description.

        Args:
            user_prompts (list or str): Source text generated by the user.
            pred (str): Candidate sentence to evaluate.
            task (str, optional): Definition of the task.
            aspect (str or list of str optional): (List of) Evaluation criterion codes.
            custom_prompt (dict, optional): Custom prompt template. Defaults to None.

        Returns:
            float: Score for the candidate sentence.
        """
        # prompts and predictions must be either a list of string or a string
        # convert to list if string
        assert isinstance(user_prompts, str) or check_list_type(
            user_prompts, str
        ), "user_prompts must be either a list of string or a string."
        assert isinstance(predictions, str) or check_list_type(
            predictions, str
        ), "predictions must be either a list of string or a string."
        assert type(predictions) == type(
            user_prompts
        ), "user_prompts and predictions must be of the same type."

        if isinstance(
            user_prompts, str
        ):  # all input variables shuold be strings in this case
            user_prompts = [user_prompts]
            predictions = [predictions]

        assert isinstance(task, str) or task is None, "task must be a string or None."
        assert custom_prompt is None or isinstance(
            custom_prompt, dict
        ), "custom_prompt must be a dict."
        assert (
            isinstance(aspect, str) or check_list_type(aspect, str) or aspect is None
        ), "aspect must be a string or a list of string or None."
        if isinstance(aspect, str):
            aspect = [aspect]

        # set aspect
        if aspect:
            for asp in aspect:
                assert (
                    asp in self.aspects.keys()
                ), "aspect is not in the list of criteria."

        # check if custom_prompt is given
        if not custom_prompt:
            assert (
                task and aspect
            ), "task and aspect must be given if no custom_prompt is given."
        if not (task and aspect):
            assert (
                custom_prompt
            ), "custom_prompt must be given if task and aspect are not given."

        # get scores accordingly
        if custom_prompt:
            scores = {
                custom_prompt["name"]: self.get_score(
                    prompts=self.get_prompt(
                        user_prompts, predictions, task, aspect, custom_prompt
                    )
                )
            }
        else:
            scores = {
                asp: self.get_score(
                    prompts=self.get_prompt(
                        user_prompts, predictions, task, asp, custom_prompt
                    )
                )
                for asp in aspect  # score for each aspect gave as input
            }

        return scores


class GPTScore:
    # pylint: disable=f-string-without-interpolation
    def __init__(
        self,
        model=None,
    ):
        """
        This class implements the GPTScore evaluation metric for generative language models.
        It is inspired by the GPTScore metric proposed in https://arxiv.org/pdf/2302.04166.pdf.
        The GPTScore from the paper is always gonna be calculated as the average log-likelihood
        of the tokens in the sentence.
        However, since the probability of each token is always gonna be between 0 and 1,
        the average log-likelihood is always gonna be negative.
        Thus, the bigger the GPTScore, the better the sentence.
        The GPTScore is always gonna be negative.

        Args:
            model (LangChain BaseChatModel): model used for evaluation. If None, the model used\
                is "gpt-3.5-turbo" by default.
        """

        assert model is None or isinstance(
            model, BaseChatModel
        ), f"model must be a LangChain BaseChatModel. model is of type {type(model).__name__}"

        self.criteria = {
            "summ": {
                "FAC": f"Generate a summary with consistent facts for the following text::\nSource:\n{{src}}:\nSummary:\n{{pred}}. This is a factually consistent summary.",
                "COV": f"Generate a summary with as much semantic coverage as possible for the following text::\nSource:\n{{src}}\nSummary:\n{{pred}}. This is a semantically comprehensive summary.",
                "CON": f"Generate factually consistent summary for the following text::\nSource:\n{{src}}\nSummary:\n{{pred}}. This is a factually consistent summary.",
                "INF": f"Generate an informative summary that captures the key points of the following text::\nSource:\n{{src}}\nSummary:\n{{pred}}. This is an informative summary.",
                "COH": f"Generate a coherent summary for the following text::\nSource:\n{{src}}\nSummary:\n{{pred}}. This is a coherent summary.",
                "REL": f"Generate a relevant summary with consistent details for the following text::\nSource:\n{{src}}\nSummary:\n{{pred}}. This is a relevant summary.",
                "FLU": f"Generate a fluent and grammatical summary for the following text::\nSource:\n{{src}}\nSummary:\n{{pred}}. This is a fluent summary.",
            },
            "machine_translation": {
                "ACC": f"Translate the following text with its core information and consistent facts:\nSource:\n{{src}}\nTranslation:\n{{pred}}. This is a factually consistent translation.",
                "FLU": f"Translate the following text to make it more grammatical and well-written:\nSource:\n{{src}}\nTranslation:\n{{pred}}. This is a fluent translation.",
                "MQM": f"Translate the following text into high-quality text with its core information:\nSource:\n{{src}}\nTranslation:\n{{pred}}. This is a high-quality translation.",
            },
            "data_to_text": {
                "INF": f"Convert the following text to another expression that preserves key information:\nSource:\n{{src}}\nConversion:\n{{pred}}. This is an informative conversion.",
                "NAT": f"Convert the following text into another expression that is human-like and natural:\nSource:\n{{src}}\nConversion:\n{{pred}}. This is a natural conversion.",
                "FLU": f"Convert the following text into another expression that preserves key information and is human-like and natural:\nSource:\n{{src}}\nConversion:\n{{pred}}. This is a fluent conversion.",
            },
            "diag": {
                "COH": f"Answer the question based on the conversation between a human and AI.\nQuestion: Is the AI coherent and maintains a good conversation flow throughout the conversation? (a) Yes. (b) No.\nConversation:\nUser: {{src}}\nAI: {{pred}}\nAnswer: Yes.",
                "DIV": f"Answer the question based on the conversation between a human and AI.\nQuestion: Is there diversity in the AI responses? (a) Yes. (b) No.\nConversation:\nUser: {{src}}\nAI: {{pred}}\nAnswer: Yes.",
                "FLE": f"Answer the question based on the conversation between a human and AI.\nQuestion: Is the AI flexible and adaptable to human and their interests? (a) Yes. (b) No.\nConversation:\nUser: {{src}}\nAI: {{pred}}\nAnswer: Yes.",
                "UND": f"Answer the question based on the conversation between a human and AI.\nQuestion: Does the AI seem to understand the human? (a) Yes. (b) No.\nConversation:\nUser: {{src}}\nAI: {{pred}}\nAnswer: Yes.",
                "INQ": f"Answer the question based on the conversation between a human and AI.\nQuestion: Is the AI inquisitive throughout the conversation? (a) Yes. (b) No.\nConversation:\nUser: {{src}}\nAI: {{pred}}\nAnswer: Yes.",
                "CON": f"Answer the question based on the conversation between a human and AI.\nQuestion: Are the responses of AI consistent in the information it provides throughout the conversation? (a) Yes. (b) No.\nConversation:\nUser: {{src}}\nAI: {{pred}}\nAnswer: Yes.",
                "INF": f"Answer the question based on the conversation between a human and AI.\nQuestion: Are the responses of AI informative throughout the conversation? (a) Yes. (b) No.\nConversation:\nUser: {{src}}\nAI: {{pred}}\nAnswer: Yes.",
                "LIK": f"Answer the question based on the conversation between a human and AI.\nQuestion: Does the AI display a likeable personality? (a) Yes. (b) No.\nConversation:\nUser: {{src}}\nAI: {{pred}}\nAnswer: Yes.",
                "DEP": f"Answer the question based on the conversation between a human and AI.\nQuestion: Does the AI discuss topics in depth? (a) Yes. (b) No.\nConversation:\nUser: {{src}}\nAI: {{pred}}\nAnswer: Yes.",
                "ERR": f"Answer the question based on the conversation between a human and AI.\nQuestion: Is the AI able to recover from errors that it makes? (a) Yes. (b) No.\nConversation:\nUser: {{src}}\nAI: {{pred}}\nAnswer: Yes.",
            },
        }
        self.tasks = self.criteria.keys()
        self.aspects = list(
            {aspect for task in self.tasks for aspect in self.criteria[task]}
        )

        if model:
            self.model = model
        else:  # Use GPT 3.5 by default if no other model provided
            self.model = get_langchain_gpt_model()

    def add_criterion(self, task: str, code: str, desc: str):
        """
        This method adds a criterion to the list of pre-defined criteria.
        Please try to follow the following example pattern to ensure consistency.

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
            task (str): Task name. (Example: "diag")
            code (str): Aspect code. (Example: "COH")
            desc (str): Aspect description.
        """

        assert isinstance(task, str), "task must be a string."
        assert isinstance(code, str), "code must be a string."
        assert isinstance(desc, str), "prompt must be a string."

        assert "{pred}" in desc  # Check that the template has the correct placeholder
        assert "{src}" in desc  # Check that the template has the correct placeholder

        # Add the criterion to the list of criteria
        if task not in self.criteria.keys():
            self.criteria[task] = {}
        self.criteria[task][code] = desc

        # Update the aspects and tasks lists
        self.tasks = self.criteria.keys()
        self.aspects = list(
            {aspect for task in self.tasks for aspect in self.criteria[task]}
        )

    def get_prompt(
        self,
        aspect: str,
        task: str,
        prompts: list,
        predictions: list,
        custom_prompt: dict = None,
    ):
        """
        This method returns a prompt template given a task description, and an aspect to evaluate.

        Args:
            prompts (str): list of source texts.
            pred (str): list of candidate sentences.
            aspect (str): Aspect to evaluate.
            task (str): Task description.
            custom_prompt (dict): Custom prompt template. Defaults to None.\
                Must contain the following keys: "task", "aspect".

        Returns:
            list: (list of) Prompt templates.
        """
        # define aspect and task
        if aspect and task:
            assert (
                aspect in self.criteria[task]
            ), f"Aspect {aspect} is not available for task {task}."
            assert self.criteria[task][
                aspect
            ], f"Prompt template for aspect {aspect} and task {task} is non-existent. Please specify a prompt template."

        # set general template
        template = (
            self.criteria[task][aspect] if (aspect and task) else str(custom_prompt)
        )

        # get final prompt for each pair of prompt and candidate sentence
        messages = []
        for prompt, prediction in zip(prompts, predictions):

            template = template.replace("{src}", prompt)
            template = template.replace("{pred}", prediction)

            message = [
                ("system", "Explain your reasoning in the following statement."),
                ("human", template),
            ]
            messages.append(message)

        return messages

    def get_score(self, prompts: list):
        """
        This method returns the GPTScore given a prompt template.

        Args:
            prompt (list): list of Prompt templates.
        Returns:
            float: GPTScore of the candidate sentence.
        """
        scores = []
        for prompt in prompts:
            self.model = self.model.bind(logprobs=True)
            response = self.model.invoke(prompt)
            response_metadata = response.response_metadata

            logprobs = [
                entry["logprob"] for entry in response_metadata["logprobs"]["content"]
            ]

            # multiply logprob of token with weight of token
            loss = sum(logprobs)
            if len(logprobs) == 0:
                avg_loss = None
            else:
                avg_loss = loss / len(logprobs)
            scores.append(avg_loss)

        return scores

    def compute(
        self,
        user_prompts: list,
        predictions: list,
        custom_prompt: dict = None,
        aspect=None,
        task: str = None,
    ):
        """
        This method computes the GPTScore for a candidate sentence given a source text,
        a system_prompt template, a user_prompt source text, an aspect to evaluate, and a task description.

        Args:
            user_prompts (list or str): (list of) Source text generated by the user.
            pred (list or str): (list of) Candidate sentence.
            custom_prompt (dict, optional): Custom prompt template. Defaults to None.
                Must contain the following keys: "task", "aspect", "name".
            aspect (str or list, optional): (List of) Aspect(s) to evaluate. Defaults to None.
            task (str, optional): Task description. Defaults to None.

        Returns:
            dict: (list of) Score for (each of) the candidate sentence per aspect.
        """
        # prompts and predictions must be either a list of string or a string
        # convert to list if string
        assert isinstance(user_prompts, str) or check_list_type(
            user_prompts, str
        ), "Source must be a either a list of string or a string."
        assert isinstance(predictions, str) or check_list_type(
            predictions, str
        ), "Pred must be a either a list of string or a string."
        assert type(predictions) == type(
            user_prompts
        ), "Source and pred must be of the same type."
        if isinstance(
            user_prompts, str
        ):  # all input variables shuold be strings in this case
            user_prompts = [user_prompts]
            predictions = [predictions]

        # If prompt is given, check that it is a dict with the following keys: "task", "aspect"
        if custom_prompt:
            assert isinstance(custom_prompt, dict), "prompt must be a dict."
            assert not aspect, "aspect must not be given if prompt is given."
            assert not task, "aspect must not be given if prompt is given."
            assert (
                "task" in custom_prompt.keys() and "aspect" in custom_prompt.keys()
            ), "prompt must contain the following keys: 'task', 'aspect'"
            assert isinstance(custom_prompt["task"], str), "task must be a string."
            assert isinstance(custom_prompt["aspect"], str), "aspect must be a string."
        else:
            # If prompt is not given, check that task and aspect are given
            assert aspect, "Aspect must be given if prompt is not given."
            assert task, "Task must be given if prompt is not given."

        # If aspect is given, check that it is a string, convert to list if string
        if aspect:
            assert isinstance(aspect, str) or check_list_type(
                aspect, str
            ), "Aspect must be either a list of string or a string."
            if isinstance(aspect, str):
                aspect = [aspect]
            for asp in aspect:
                assert asp in self.aspects, f"Aspect must be one of {self.aspects}."

        # If task is given, check that it is a string
        if task:
            assert isinstance(task, str), "Task must be a string."
            assert task in self.tasks, f"Task must be one of {self.tasks}."

        # Generative LLM is given a prompt template and some context information
        # TODO: with custom_prompt only one aspect can be evaluated at a time for now
        if custom_prompt:
            scores = {
                custom_prompt["aspect"]: self.get_score(
                    prompts=self.get_prompt(
                        aspect, task, user_prompts, predictions, custom_prompt
                    )
                )
            }
        else:
            scores = {
                asp: self.get_score(
                    prompts=self.get_prompt(
                        asp, task, user_prompts, predictions, custom_prompt
                    )
                )
                for asp in aspect  # score for each aspect gave as input
            }

        return scores


class Relevance:
    def __init__(self, llm=None) -> None:
        """
        This class implements the relevance evaluation metric for generative language models.
        The relevance metric evaluates if the submission refers to or accurately conveys the\
        information from the input text,
        even if it is not an exact quote.
        
        Args:
            llm (LangChain BaseLanguageModel): model used for evaluation. If None,\
                the model is chosen as "gpt-4" by default.
        """
        assert llm is None or isinstance(
            llm, BaseLanguageModel
        ), f"llm must be a LangChain BaseLanguageModel. model is of type {type(llm).__name__}"

        self.llm = llm if llm else None
        self.criterion = {
            "relevance": "Does the submission refer to or accurately convey the information from the input text, \
                even if it is not an exact quote?"
        }
        self.evaluator = load_evaluator(
            "criteria", criteria=self.criterion, llm=self.llm
        )

    def compute(self, user_prompts: list, predictions: list):
        """
        This method computes the relevance score for a candidate sentence given a source text.
        In other words, it validates that the candidate sentence (response) is related to the query topic,
        and meets the query requirements.

        Args:
            user_prompts (list): Source text generated by the user.
            pred (list): Candidate sentence.

        Returns:
            dict: Relevance score for the candidate sentence. The dictionary contains the following keys:

                - score (int): Relevance score. Binary integer value (0 or 1), where 1 indicates that the sentence is\
                    relevant and 0 indicates that the sentence is irrelevant.
                - value (str): Relevance value. Y or N, where Y indicates that the sentence is relevant and N indicates\
                    that the sentence is irrelevant.
                - reasoning (str): Reasoning for the relevance score.
        """
        assert isinstance(user_prompts, list), "user_prompts must be a list."
        assert isinstance(predictions, list), "predictions must be a list."
        assert all(
            isinstance(prompt, str) for prompt in user_prompts
        ), "All elements in user_prompts must be strings."
        assert all(
            isinstance(pred, str) for pred in predictions
        ), "All elements in predictions must be strings."

        result = self.evaluator.evaluate_strings(
            prediction=predictions, input=user_prompts
        )

        return result


class Correctness:
    def __init__(self, llm=None) -> None:
        """
        This class implements the correctness evaluation metric for generative language models.
        The correctness metric evaluates if the submission is correct, accurate, and factual.
        This definition is based on LangChain's `labeled_criteria evaluator \
            <https://python.langchain.com/v0.2/api_reference/_modules/langchain/evaluation/criteria/eval_chain.html#Criteria>`_.
        
        Args:
            llm (LangChain BaseLanguageModel): model used for evaluation. If None,\
                the model is chosen as "gpt-4" by default.
        """
        assert llm is None or isinstance(
            llm, BaseLanguageModel
        ), f"llm must be a LangChain BaseLanguageModel. model is of type {type(llm).__name__}"

        self.llm = llm if llm else None
        self.evaluator = load_evaluator(
            "labeled_criteria", criteria="correctness", llm=self.llm
        )

    def compute(self, user_prompts: list, predictions: list, references: list):
        """
        This method computes the correctness score for a candidate sentence given a source text and a reference.

        Args:
            user_prompts (list): Source text generated by the user.
            pred (list): Candidate sentence.
            references (list): Reference sentence.
        Returns:
            dict: Correctness score for the candidate sentence. The dictionary contains the following keys:

                - score (int) : Correctness score. Binary integer value (0 or 1), where 1 indicates that the sentence\
                    is correct and 0 indicates that the sentence is incorrect.
                - value (str) : Correctness value. Y or N, where Y indicates that the sentence is correct and N\
                    indicates that the sentence is incorrect.
                - reasoning (str) : Reasoning for the correctness score.
        """
        assert isinstance(user_prompts, list), "user_prompts must be a list."
        assert isinstance(predictions, list), "predictions must be a list."
        assert isinstance(references, list), "references must be a list."
        assert all(
            isinstance(prompt, str) for prompt in user_prompts
        ), "All elements in user_prompts must be strings."
        assert all(
            isinstance(pred, str) for pred in predictions
        ), "All elements in predictions must be strings."
        assert all(
            isinstance(ref, str) for ref in references
        ), "All elements in references must be strings."

        result = self.evaluator.evaluate_strings(
            prediction=predictions, input=user_prompts, reference=references
        )

        return result


class Faithfulness:
    def __init__(self, llm=None) -> None:
        """
        This class implements the faithfulness evaluation metric for generative language models.
        The faithfulness metric evaluates if the submission contains information not present in the input or reference.

        Args:
            llm (LangChain BaseLanguageModel): model used for evaluation. If None,\
                the model is chosen as "gpt-4" by default.
        """
        assert llm is None or isinstance(
            llm, BaseLanguageModel
        ), f"llm must be a LangChain BaseLanguageModel. model is of type {type(llm).__name__}"

        self.llm = llm if llm else None
        self.criterion = {
            "faithfulness": "Does this submission contain information not present in the input or reference?",
        }
        self.evaluator = load_evaluator(
            "labeled_criteria", criteria=self.criterion, llm=self.llm
        )

    def compute(self, user_prompts: list, predictions: list, references: list):
        """
        This method computes the faithfulness score for a candidate sentence given a source text and a reference.

        Args:
            user_prompts (list): Source text generated by the user.
            pred (list): Candidate sentence.
            references (list): Reference sentence.
        Returns:
            dict: Faithfulness score for the candidate sentence. The dictionary contains the following keys:

                - score (int) : Faithfulness score. Binary integer value (0 or 1), where 1 indicates that\
                    the sentence is faithful and 0 indicates that the sentence is not faithful\
                    (i.e. it contains hallucinations).
                - value (str) : Faithfulness value. Y or N, where Y indicates that the sentence is faithful and\
                    N indicates that the sentence is not faithful.
                - reasoning (str): Reasoning for the faithfulness score.
        """
        assert isinstance(user_prompts, list), "user_prompts must be a list."
        assert isinstance(predictions, list), "predictions must be a list."
        assert isinstance(references, list), "references must be a list."
        assert all(
            isinstance(prompt, str) for prompt in user_prompts
        ), "All elements in user_prompts must be strings."
        assert all(
            isinstance(pred, str) for pred in predictions
        ), "All elements in predictions must be strings."
        assert all(
            isinstance(ref, str) for ref in references
        ), "All elements in references must be strings."

        result = self.evaluator.evaluate_strings(
            prediction=predictions, input=user_prompts, reference=references
        )

        return result


class NegativeRejection:
    def __init__(self, llm=None) -> None:
        """
        This class implements the negative rejection evaluation metric for generative language models.
        The negative rejection metric evaluates if the submission refuses to answer when the answer\
            is not present in the input or reference.

        Args:
            llm (LangChain BaseLanguageModel): model used for evaluation. If None,\
                the model is chosen as "gpt-4" by default.
        """
        assert llm is None or isinstance(
            llm, BaseLanguageModel
        ), f"llm must be a LangChain BaseLanguageModel. model is of type {type(llm).__name__}"

        self.llm = llm if llm else None
        self.criterion = {
            "negative_rejection": "Does this submission refuse to answer when the answer is not present in the input or reference?",
        }
        self.evaluator = load_evaluator(
            "labeled_criteria", criteria=self.criterion, llm=self.llm
        )

    # measures the ability of the system to refuse to andwer in the absence of evidences (F1, exact match, LLM as judge)
    def compute(self, user_prompts: list, predictions: list, references: list):
        """
        This class computes the ability of the system to refuse to answer in the absence of evidence.

        Args:
            user_prompts (list): Source text generated by the user.
            pred (list): Candidate sentence.
            references (list): Reference sentence.

        Returns:
            dict: Negative rejection score for the candidate sentence.\
                The dictionary contains the following keys:

                - score (int): Negative rejection score. Binary integer value (0 or 1), where 1 indicates\
                    that the sentence is a refusal to answer and 0 indicates that the sentence\
                        is not a refusal to answer.
                - value (str): Negative rejection value. Y or N, where Y indicates\
                    that the sentence is a refusal to answer and N indicates that the sentence\
                        is not a refusal to answer.
                - reasoning (str): Reasoning for the negative rejection score.
        """
        assert isinstance(user_prompts, list), "user_prompts must be a list."
        assert isinstance(predictions, list), "predictions must be a list."
        assert isinstance(references, list), "references must be a list."
        assert all(
            isinstance(prompt, str) for prompt in user_prompts
        ), "All elements in user_prompts must be strings."
        assert all(
            isinstance(pred, str) for pred in predictions
        ), "All elements in predictions must be strings."
        assert all(
            isinstance(ref, str) for ref in references
        ), "All elements in references must be strings."
        assert len(predictions) == len(
            references
        ), "Predictions and references must be of the same length."

        result = self.evaluator.evaluate_strings(
            prediction=predictions, input=user_prompts, reference=references
        )

        return result


class HallucinationScore:
    def compute(self, predictions: list, references: list):
        """
        This method computes the hallucination scores for a candidate sentence given a reference sentence.

        Args:
            predictions (list): Candidate sentences (e.g., model outputs).
            references (list): Reference sentences (e.g., ground truth).

        Returns:
            dict: Hallucination detection score. The dictionary contains the following keys:

                - f1_score (float): F1 score, representing the overlap between the prediction and the reference.
                - exact_match (int): Binary integer value (0 or 1), where 1 indicates that the prediction exactly\
                    matches the reference and 0 indicates it does not.
        """
        assert isinstance(predictions, list), "predictions must be a list."
        assert isinstance(references, list), "references must be a list."
        assert len(predictions) == len(
            references
        ), "Predictions and references must be of the same length."
        assert all(
            isinstance(pred, str) for pred in predictions
        ), "All elements in predictions must be strings."
        assert all(
            isinstance(ref, str) for ref in references
        ), "All elements in references must be strings."

        total_f1 = 0.0
        exact_matches = 0

        for pred, ref in zip(predictions, references):
            # Compute token-based F1 score
            pred_tokens = set(pred.split())
            ref_tokens = set(ref.split())

            common_tokens = pred_tokens.intersection(ref_tokens)
            precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
            recall = len(common_tokens) / len(ref_tokens) if ref_tokens else 0

            f1_ = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            total_f1 += f1_

            # Compute exact match
            if pred == ref:
                exact_matches += 1

        avg_f1 = total_f1 / len(predictions) if predictions else 0.0
        exact_match_score = exact_matches / len(predictions) if predictions else 0.0

        result = {"f1_score": avg_f1, "exact_match": int(exact_match_score == 1)}

        return result
