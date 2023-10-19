import openai
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


class GPTScore:
    def __init__(self, model="gpt2"):
        """
        GPTScore is a metric which allows to evaluate generative models on a variety of tasks.
        GPTScore(h|d, a, S) =  sum_{t=1}^m w_t * log p(h_t | h_{<t}, T(d, a, S), theta)
        where w_t is a weight assigned to each token h_t.
        T is a prompt template with
        d: task description,
        a: aspect to evaluate,
        S: context information.
        and theta are model parameters.
        GPTScore does not require any reference text.

        Args:
            model (str, optional): Model name. Defaults to "gpt2".
        """
        self.model = model
        self.huggingface_models = [
            "meta-llama/Llama-2-7b-chat-hf",
            "gpt2",
            "mistralai/Mistral-7B-v0.1",
        ]
        self.openai_models = ["gpt-3.5-turbo"]
        self.aspects = [
            "COV",
            "FAC",
            "FLU",
            "CON",
            "INF",
            "COH",
            "REL",
            "ACC",
            "MQM",
            "INT",
            "ENG",
            "SPE",
            "COR",
            "SEM",
            "UND",
            "ERR",
            "DIV",
            "DEP",
            "LIK",
            "FLE",
            "INQ",
        ]
        self.models = [
            "meta-llama/Llama-2-7b-chat-hf",
            "gpt-3.5-turbo",
            "gpt2",
            "mistralai/Mistral-7B-v0.1",
        ]
        self.tasks = ["summ", "MT", "D2T", "diag"]

        assert isinstance(model, str), "Model must be a string."
        assert model in self.models, f"Model must be one of {self.models}."

    def huggingface_logprobs(self, prompts):
        """
        This method outputs the log-likelihood of the next token given the previous tokens and the prompt template.
        Args:
            prompts (list of str): List of prompt templates.
        Returns:
            list: List of log-likelihoods for each candidate sentence.
        """

        tokenizer = AutoTokenizer.from_pretrained(self.model)
        llm = AutoModelForCausalLM.from_pretrained(self.model)
        inputs = tokenizer(prompts, return_tensors="pt")

        outputs = llm.generate(
            **inputs,
            max_new_tokens=50,
            return_dict_in_generate=True,
            output_scores=True,
        )

        logprobs = np.array(
            llm.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            ).tolist()
        )

        return logprobs

    def openai_logprobs(self, prompts, api_key):
        """
        This method outputs the log-likelihood of the next token given the previous tokens and the prompt template.
        Args:
            prompts (list of str): List of prompt templates.
            api_key (str): OpenAI API key.
        Returns:
            list: List of log-likelihoods for each candidate sentence.
        """
        openai.api_key = api_key
        outputs = openai.Completion.create(
            model=self.model,
            prompt=prompts,
            logprobs=5,
        )

        logprobs = outputs["choices"][0]["logprobs"]

        return logprobs

    def get_prompts(self, aspect, task, sources, preds):
        """
        This method returns a list of prompt templates given a task description, and an aspect to evaluate.
        Args:
            aspect (str): Aspect to evaluate.
            task (str): Task description.
            sources (list of str): Source texts.
            preds (list of str): Candidate sentences.
        Returns:
            list: List of prompt templates.
        """
        return [
            self.get_prompt(aspect, task, src, pred)
            for (src, pred) in zip(sources, preds)
        ]

    def get_prompt(self, aspect, task, src, pred):
        """
        This method returns a prompt template given a task description, and an aspect to evaluate.
        Args:
            aspect (str): Aspect to evaluate.
            task (str): Task description.
            src (str): Source text.
            pred (str): Candidate sentence.
        Returns:
            str: Prompt template.
        """

        templates = {
            "summ": {
                "FAC": f"Generate a summary with consistent facts for the following text: {src}\n\nTl;dr{pred}",
                "COV": f"Generate a summary with as much semantic coverage as possible for the following text: {src}\n\nTl;dr{pred}",
                "CON": f"Generate factually consistent summary for the following text: {src}\n\nTl;dr{pred}",
                "INF": f"Generate an informative summary that captures the key points of the following text:{src}\n\nTl;dr{pred}",
                "COH": f"Generate a coherent summary for the following text: {src}\n\nTl;dr{pred}",
                "REL": f"Generate a relevant summary with consistent details for the following text: {src}\n\nTl;dr{pred}",
                "FLU": f"Generate a fluent and grammatical summary for the following text: {src}\n\nTl;dr{pred}",
            },
            "MT": {
                "ACC": f"Rewrite the following text with its core information and consistent facts:{src} In other words, {pred}",
                "FLU": f"Rewrite the following text to make it more grammatical and well-written:{src} In other words,{pred}",
                "MQM": f"Rewrite the following text into high-quality text with its core information:{src} In other words,{pred}",
            },
            "D2T": {
                "INF": f"Convert the following text to another expression that preserves key information:\n\n{src} In other words, {pred}",
                "NAT": f"Convert the following text into another expression that is human-like and natural:\n\n{src} In other words, {pred}",
                "FLU": f"Convert the following text into another expression that preserves key information and is human-like and natural:\n\n{src} In other words, {pred}",
            },
            "diag": {
                "COH": f"Answer the question based on the conversation between a human and AI.\nQuestion: Is the AI coherent and maintains a good conversation flow throughout the conversation? (a) Yes. (b) No.\nConversation: {src + pred}\nAnswer: Yes.",
                "DIV": f"Answer the question based on the conversation between a human and AI.\nQuestion: Is there diversity in the AI responses? (a) Yes. (b) No.\nConversation: {src + pred}\nAnswer: Yes.",
                "FLE": f"Answer the question based on the conversation between a human and AI.\nQuestion: Is the AI flexible and adaptable to human and their interests? (a) Yes. (b) No.\nConversation: {src + pred}\nAnswer: Yes.",
                "UND": f"Answer the question based on the conversation between a human and AI.\nQuestion: Does the AI seem to understand the human? (a) Yes. (b) No.\nConversation: {src + pred}\nAnswer: Yes.",
                "INQ": f"Answer the question based on the conversation between a human and AI.\nQuestion: Is the AI inquisitive throughout the conversation? (a) Yes. (b) No.\nConversation: {src + pred}\nAnswer: Yes.",
                "CON": f"Answer the question based on the conversation between a human and AI.\nQuestion:  Are the responses of AI consistent in the information it provides throughout the conversation? (a) Yes. (b) No.\nConversation: {src + pred}\nAnswer: Yes.",
                "INF": f"Answer the question based on the conversation between a human and AI.\nQuestion: Are the responses of AI informative throughout the conversation? (a) Yes. (b) No.\nConversation: {src + pred}\nAnswer: Yes.",
                "LIK": f"Answer the question based on the conversation between a human and AI.\nQuestion:  Does the AI display a likeable personality? (a) Yes. (b) No.\nConversation: {src + pred}\nAnswer: Yes.",
                "DEP": f"Answer the question based on the conversation between a human and AI.\nQuestion: Does the AI discuss topics in depth? (a) Yes. (b) No.\nConversation: {src + pred}\nAnswer: Yes.",
                "ERR": f"Answer the question based on the conversation between a human and AI.\nQuestion: Is the AI able to recover from errors that it makes? (a) Yes. (b) No.\nConversation: {src + pred}\nAnswer: Yes.",
            },
        }

        # Check that the corresponding entry exists in the prompt template
        assert (
            aspect in templates[task]
        ), f"Aspect {aspect} is not available for task {task}."
        # Check that the prompt template is not empty
        assert templates[task][
            aspect
        ], f"Prompt template for aspect {aspect} and task {task} is non-existent. Please specify a prompt template."

        return templates[task][aspect]

    def compute(
        self, sources, preds, prompts=None, aspect=None, task=None, api_key=None
    ):
        """
        This method computes GPTScore for a list of candidate sentences given a task description,
        an aspect to evaluate and context information.
        The possible values for aspect are:
        - (COV): Semantic coverage. How many semantic content units from the reference text
                    are covered by the generated text?
        - (FAC): Factuality. Does the generated text preserve the factual statements of the source text?)
        - (FLU): Fluency. Is the generated text well-written and grammatical?
        - (CON): Consistency. Is the generated text consistent in the information it provides?
        - (INF): Informativeness. How well does the generated text capture the key ideas of its source text?
        - (COH): Coherence. How much does the generated text make sense?
        - (REL): Relevance. How well is the generated text relevant to its source text?
        - (ACC): Accuracy. Are there inaccuracies, missing, or unfactual content in the generated text?
        - (MQM): Multidimensional MT How is the overall quality of the generated text?
        - (INT): Interest. Is the generated text interesting?
        - (ENG): Engagement. Is the generated text engaging?
        - (SPE): Specific. Is the generated text generic or specific to the source text?
        - (COR): Correctness. Is the generated text correct or was there a misunderstanding of the source text?
        - (SEM): Semantically appropriate. Is the generated text semantically appropriate?
        - (UND): Understandability. Is the generated text understandable?
        - (ERR): Error Recovery. Is the system able to recover from errors that it makes?
        - (DIV): Diversity. Is there diversity in the system responses?
        - (DEP): Depth. Does the system discuss topics in depth?
        - (LIK): Likeability. Does the system display a likeable personality?
        - (FLE): Flexibility. Is the system flexible and adaptable to the user and their interests?
        - (INQ): Inquisitiveness. Is the system inquisitive throughout the conversation?

        Possible tasks are for pre-made prompts are:
        - (summ): Summarization. Generating an informative and fluent summary for a given long text.
        - (MT): Machine Translation. Translate a sentence from one language to another.
        - (D2T): Data to Text. Automatically generate a fluent and factual description for a given table.
        - (diag): Dialogue. Generate an engaging and informative response based on the dialogue history.

        Args:
            sources (list of str): Source texts.
            preds (list of str): Candidate sentences.
            prompts (str): Prompt template. If None, a default prompt template is used.
            aspect (list): List of aspects to evaluate.
            task (str): Task description.
            api_key (str): OpenAI API key.

        Returns:
            list: List of scores for each candidate sentence.
        """
        assert isinstance(sources, list) and isinstance(
            sources[0], str
        ), "Source must be a list of strings."
        assert isinstance(preds, list) and isinstance(
            preds[0], str
        ), "Prediction must be a list of strings."

        assert isinstance(self.model, str), "Model must be a string."
        assert self.model in self.models, f"Model must be one of {self.models}."

        # If prompt is given, check that it is a list of string
        if prompts:
            assert isinstance(prompts, list) and isinstance(
                prompts[0], str
            ), "Prompts must be a list of strings."
            assert not aspect, "Aspect must not be given if prompt is given."
            assert not task, "Task must not be given if prompt is given."
        else:
            # If prompt is not given, check that task and aspect are given
            assert aspect, "Aspect must be given if prompt is not given."
            assert task, "Task must be given if prompt is not given."

        # If aspect is given, check that it is a string
        if aspect:
            assert isinstance(aspect, str), "Aspect must be a string."
            assert aspect in self.aspects, f"Aspect must be one of {self.aspects}."

        # If task is given, check that it is a string
        if task:
            assert isinstance(task, str), "Task must be a string."
            assert task in self.tasks, f"Task must be one of {self.tasks}."

        # Generative LLM is given a prompt template and some context information
        prompts = prompts or self.get_prompts(aspect, task, sources, preds)

        # Model predicts log-likelihood of the next token given the previous tokens and the prompt template
        if self.model in self.huggingface_models:
            logprobs = self.huggingface_logprobs(prompts)

        elif self.model in self.openai_models:
            logprobs = self.openai_logprobs(prompts, api_key)

        # Compute GPTScores
        scores = []
        for i, pred in enumerate(preds):
            pred_tokens = pred.split()
            pred_logprobs = logprobs[i][: len(pred_tokens)]
            scores.append(np.mean(pred_logprobs))

        return scores
