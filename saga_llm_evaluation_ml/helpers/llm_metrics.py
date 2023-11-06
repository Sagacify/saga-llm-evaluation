import numpy as np
from huggingface_hub import hf_hub_download
from llama_cpp import Llama


class SelfCheckGPT:
    def __init__(
        self,
        model,
        eval_model=False,
        eval_model_name_or_path="TheBloke/Llama-2-7b-Chat-GGUF",
        eval_model_basename="llama-2-7b-chat.Q4_K_M.gguf",
    ):
        """
        This class implements the self-check GPT evaluation metric for generative language models.
        It is inspired by the self-check metric proposed in https://arxiv.org/pdf/2303.08896.pdf.
        Args:
            model (transformers.PreTrainedModel): GPT model to evaluate.
            eval_model (LLama model, optional): Evaluation model. If False, the evaluation model is
            downloaded from the HuggingFace Hub.
            eval_model_name_or_path (str): Evaluation model name or path. Defaults to "TheBloke/Llama-2-7b-Chat-GGUF".
            eval_model_basename (str): Evaluation model basename. Defaults to "llama-2-7b-chat.Q4_K_M.gguf".
        """
        assert isinstance(
            eval_model_name_or_path, str
        ), "eval_model_name_or_path must be a string."
        assert isinstance(
            eval_model_basename, str
        ), "eval_model_basename must be a string."

        self.model = model
        if not eval_model:
            self.eval_model_path = hf_hub_download(
                repo_id=eval_model_name_or_path, filename=eval_model_basename
            )

            self.eval_model = Llama(
                model_path=self.eval_model_path, n_threads=2, verbose=False  # CPU cores
            )
        else:
            self.eval_model = eval_model

    def get_prompt(self, pred, sample, question):
        """
        This method returns a prompt template given a candidate sentence, a sample sentence, and a question.
        Args:
            pred (str): Candidate sentence.
            sample (str): Sample sentence.
            question (str): Question asked to the model for which it generated $pred.

        Returns:
            str: Prompt template.
        """
        system_prompt = "You are a helpful, polite and concise assistant. Your task is to check if two texts provide the same answer to a given question. Always answer with a single word. The possible answers are either YES or NO.\n\n"
        question = "###Question:\n" + question
        text1 = "\n###Text 1: " + sample
        text2 = "\n###Text 2: " + pred

        prompt_template = f"""SYSTEM: {system_prompt}
        USER: {question + text1 + text2}
        ASSISTANT (YES or NO):"""

        return prompt_template

    def get_prompts(self, pred, samples, question):
        """
        This method returns a list of prompt templates given a candidate sentence, a list
        of sample sentences, and a question.
        Args:
            pred (str): Candidate sentence.
            samples (list of str): List of sample sentences.
            question (str): Question asked to the model for which it generated $pred.

        Returns:
            list: List of prompt templates.
        """
        return [self.get_prompt(pred, sample, question) for sample in samples]

    def compute(self, question, pred, n_samples):
        """
        Args:
            question (str): Question asked to the model for which it generated $pred.
            pred (str): Candidate sentence.
            n_samples (int): Number of samples to generate.

        Returns:
            score (float): Score for the candidate sentence.
        """
        assert isinstance(question, str), "Prediction must be a string."
        assert isinstance(pred, str), "Prediction must be a string."
        assert isinstance(n_samples, int), "Number of samples must be an integer."
        assert n_samples > 0, "Number of samples must be greater than 0."
        assert question and pred, "Question and prediction must be non-empty."

        # Generate n_samples samples from the model
        samples = []
        for _ in range(n_samples):
            system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible."
            prompt_template = f"""SYSTEM: {system_prompt}
            USER: {question}
            ASSISTANT:"""

            response = self.model(prompt_template, max_tokens=200)
            sample = response["choices"][0]["text"]
            samples.append(sample)

        # For each sample, ask evaluator model to evaluate the sample
        prompts = self.get_prompts(pred, samples, question)
        scores = []
        for prompt in prompts:
            answer = self.eval_model(prompt, max_tokens=200)["choices"][0]["text"]
            scores.append(answer)

        # Compute the score: how often the sentence if supported by the sample
        score = np.mean([1 if "yes" in score.lower() else 0 for score in scores])

        return score


class GEval:
    def __init__(
        self,
        model=False,
        model_name_or_path="TheBloke/Llama-2-7b-Chat-GGUF",
        model_basename="llama-2-7b-chat.Q4_K_M.gguf",
    ):
        """
        This class implements the GEval evaluation metric for generative language models.
        It is inspired by the GEval metric proposed in https://arxiv.org/pdf/2303.16634.pdf.
        Args:
            model (Llama model): model used for evaluation. If False, the model is downloaded from the HuggingFace Hub.
            model_name_or_path (str): Model name or path. Defaults to "TheBloke/Llama-2-7b-Chat-GGUF".
            model_basename (str): Model basename. Defaults to "llama-2-7b-chat.Q4_K_M.gguf".
        """
        assert isinstance(
            model_name_or_path, str
        ), "model_name_or_path must be a string."
        assert isinstance(model_basename, str), "model_basename must be a string."

        if not model:
            self.model_path = hf_hub_download(
                repo_id=model_name_or_path, filename=model_basename
            )

            self.lcpp_llm = Llama(
                model_path=self.model_path,
                n_threads=2,  # CPU cores
                logits_all=True,
                n_ctx=1000,
            )
        else:
            self.lcpp_llm = model

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

    def add_task(self, name, definition):
        """
        This method adds a task to the list of pre-defined tasks.
        Please try to follow the following example pattern to ensure consistency.
        Example:
        "summ": "You will be given one summary written for a news article. Your task is to rate the summary on one metric. Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.",

        Args:
            name (str): Task name.
            definition (str): Task description.
        """
        assert isinstance(name, str), "name must be a string."
        assert isinstance(definition, str), "definition must be a string."

        self.tasks[name] = definition

    def add_aspect(self, code, name, prompt):
        """
        This method adds an aspect to the list of pre-defined aspects.
        Please try to follow the following example pattern to ensure consistency.
        Example:
        "COH": {
            "name": "Coherence",
            "prompt": "Coherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby ”the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to sentence to a coherent body of information about a topic.”",
        },

        Args:
            code (str): Aspect code.
            name (str): Aspect name.
            prompt (str): Aspect prompt.
        """
        assert isinstance(code, str), "code must be a string."
        assert isinstance(name, str), "name must be a string."
        assert isinstance(prompt, str), "prompt must be a string."

        self.aspects[code] = {"name": name, "prompt": prompt}

    def get_prediction(self, prompt):
        """
        This method returns a prediction given a prompt template.
        Args:
            prompt (str): Prompt template.

        Returns:
            response (dict): Response from the model.
        """
        response = self.lcpp_llm.create_completion(
            prompt=prompt,
            max_tokens=250,
            temperature=0.5,
            top_p=0.95,
            logprobs=5,
            repeat_penalty=1.2,
            top_k=50,
            echo=True,
        )
        return response

    def get_cot(self, prompt):
        """
        This method returns a chain of thoughts given a prompt template.
        Args:
            prompt (str): Prompt template.

        Returns:
            cot (str): Chain of thoughts.
        """
        title = "\nEvaluation steps:\n"
        cot = self.get_prediction(prompt + title)["choices"][0]["text"]
        return cot

    # pylint: disable=consider-iterating-dictionary
    def get_prompt(self, src, pred, task, aspect, custom_prompt):
        """
        Args:
            src (str): Source text.
            pred (str): Candidate sentence to evaluate.
            task (str): Definition of the task.
            aspect (str): Evaluation criterion code.
            custom_prompt (dict): Custom prompt template.
                Must contain the following keys: "task", "aspect", "name".
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
        name = (
            self.aspects[aspect]["name"]
            if aspect in self.aspects.keys()
            else custom_prompt["name"]
        )

        prompt = f"{definition} {crit}"

        # Chain of thoughts, set of intermediate instructions generated by llm detailing evaluation steps
        auto_cot = self.get_cot(prompt)

        return (
            prompt
            + auto_cot
            + "\n Example:\n Source Text:\n"
            + src
            + "\n Generated text:\n"
            + pred
            + "\n Evaluation Form (scores ONLY):\n"
            + name
            + ": "
        )

    def get_score(self, prompt):
        """
        Args:
            prompt (str): Prompt template.

        Returns:
            score (float): Score for the candidate sentence.
        """
        response = self.get_prediction(prompt)
        tokens = response["choices"][0]["logprobs"]["tokens"]
        top_logprobs = response["choices"][0]["logprobs"]["top_logprobs"]

        # Extract evaluation form from tokens ()
        template_tokens = [
            " E",
            "valu",
            "ation",
            " Form",
            " (",
            "sc",
            "ores",
            " ON",
            "LY",
            "):",
        ]
        start_index = tokens.index(template_tokens[-1]) + 1
        # Extract number index from the remaining tokens
        for token in tokens[start_index:]:
            if token.isdigit():
                number_index = tokens.index(token)
                break

        # Get logprobs associated with number
        logprobs = top_logprobs[number_index]

        # Compute score
        # Get only keys that are numbers
        number_keys = [int(key) for key in logprobs.keys() if key.isdigit()]
        number_logprobs = [logprobs[str(key)] for key in number_keys]
        number_probs = [np.exp(logprob) for logprob in number_logprobs]

        score = np.sum(np.multiply(number_keys, number_probs)) / len(number_keys)

        return score

    def compute(self, source, pred, task=None, aspect=None, custom_prompt=None):
        """
        This method computes the GEval score for a candidate sentence given a source text,
        a prompt template, an aspect to evaluate, and a task description.
        Args:
            source (str): Source text.
            pred (str): Candidate sentence to evaluate.
            task (str, optional): Definition of the task.
            aspect (str, optional): Evaluation criterion code.
            custom_prompt (dict, optional): Custom prompt template. Defaults to None.

        Returns:
            score (float): Score for the candidate sentence.
        """
        assert isinstance(source, str), "source must be a string."
        assert isinstance(pred, str), "pred must be a string."
        assert isinstance(task, str) or task is None, "task must be a string or None."
        assert (
            isinstance(aspect, str) or aspect is None
        ), "aspect must be a string or None."
        assert custom_prompt is None or isinstance(
            custom_prompt, dict
        ), "custom_prompt must be a dict."
        if aspect:
            assert (
                aspect in self.aspects.keys()
            ), "aspect is not in the list of criteria."
        if not custom_prompt:
            assert (
                task and aspect
            ), "task and aspect must be given if no custom_prompt is given."
        if not (task and aspect):
            assert (
                custom_prompt
            ), "custom_prompt must be given if task and aspect are not given."

        prompt = self.get_prompt(source, pred, task, aspect, custom_prompt)
        return self.get_score(prompt)


class GPTScore:
    # pylint: disable=f-string-without-interpolation
    def __init__(
        self,
        model=False,
        model_name_or_path="TheBloke/Llama-2-7b-Chat-GGUF",
        model_basename="llama-2-7b-chat.Q4_K_M.gguf",
    ):
        """
        This class implements the GPTScore evaluation metric for generative language models.
        It is inspired by the GPTScore metric proposed in https://arxiv.org/pdf/2302.04166.pdf.
        Args:
            model (Llama model): model used for evaluation. If False, the model is downloaded from the HuggingFace Hub.
            model_name_or_path (str): Model name or path. Defaults to "TheBloke/Llama-2-7b-Chat-GGUF".
            model_basename (str): Model basename. Defaults to "llama-2-7b-chat.Q4_K_M.gguf".
        """
        assert isinstance(
            model_name_or_path, str
        ), "model_name_or_path must be a string."
        assert isinstance(model_basename, str), "model_basename must be a string."

        self.templates = {
            "summ": {
                "FAC": f"Generate a summary with consistent facts for the following text: {{src}}\n\nTl;dr{{pred}}",
                "COV": f"Generate a summary with as much semantic coverage as possible for the following text: {{src}}\n\nTl;dr{{pred}}",
                "CON": f"Generate factually consistent summary for the following text: {{src}}\n\nTl;dr{{pred}}",
                "INF": f"Generate an informative summary that captures the key points of the following text:{{src}}\n\nTl;dr{{pred}}",
                "COH": f"Generate a coherent summary for the following text: {{src}}\n\nTl;dr{{pred}}",
                "REL": f"Generate a relevant summary with consistent details for the following text: {{src}}\n\nTl;dr{{pred}}",
                "FLU": f"Generate a fluent and grammatical summary for the following text: {{src}}\n\nTl;dr{{pred}}",
            },
            "MT": {
                "ACC": f"Rewrite the following text with its core information and consistent facts:{{src}} In other words, {{pred}}",
                "FLU": f"Rewrite the following text to make it more grammatical and well-written:{{src}} In other words,{{pred}}",
                "MQM": f"Rewrite the following text into high-quality text with its core information:{{src}} In other words,{{pred}}",
            },
            "D2T": {
                "INF": f"Convert the following text to another expression that preserves key information:\n\n{{src}} In other words, {{pred}}",
                "NAT": f"Convert the following text into another expression that is human-like and natural:\n\n{{src}} In other words, {{pred}}",
                "FLU": f"Convert the following text into another expression that preserves key information and is human-like and natural:\n\n{{src}} In other words, {{pred}}",
            },
            "diag": {
                "COH": f"Answer the question based on the conversation between a human and AI.\nQuestion: Is the AI coherent and maintains a good conversation flow throughout the conversation? (a) Yes. (b) No.\nConversation:\nUser: {{src}}\nAI: {{pred}}\nAnswer:",
                "DIV": f"Answer the question based on the conversation between a human and AI.\nQuestion: Is there diversity in the AI responses? (a) Yes. (b) No.\nConversation:\nUser: {{src}}\nAI: {{pred}}\nAnswer:",
                "FLE": f"Answer the question based on the conversation between a human and AI.\nQuestion: Is the AI flexible and adaptable to human and their interests? (a) Yes. (b) No.\nConversation:\nUser: {{src}}\nAI: {{pred}}\nAnswer:",
                "UND": f"Answer the question based on the conversation between a human and AI.\nQuestion: Does the AI seem to understand the human? (a) Yes. (b) No.\nConversation:\nUser: {{src}}\nAI: {{pred}}\nAnswer:",
                "INQ": f"Answer the question based on the conversation between a human and AI.\nQuestion: Is the AI inquisitive throughout the conversation? (a) Yes. (b) No.\nConversation:\nUser: {{src}}\nAI: {{pred}}\nAnswer:",
                "CON": f"Answer the question based on the conversation between a human and AI.\nQuestion: Are the responses of AI consistent in the information it provides throughout the conversation? (a) Yes. (b) No.\nConversation:\nUser: {{src}}\nAI: {{pred}}\nAnswer:",
                "INF": f"Answer the question based on the conversation between a human and AI.\nQuestion: Are the responses of AI informative throughout the conversation? (a) Yes. (b) No.\nConversation:\nUser: {{src}}\nAI: {{pred}}\nAnswer:",
                "LIK": f"Answer the question based on the conversation between a human and AI.\nQuestion: Does the AI display a likeable personality? (a) Yes. (b) No.\nConversation:\nUser: {{src}}\nAI: {{pred}}\nAnswer:",
                "DEP": f"Answer the question based on the conversation between a human and AI.\nQuestion: Does the AI discuss topics in depth? (a) Yes. (b) No.\nConversation:\nUser: {{src}}\nAI: {{pred}}\nAnswer:",
                "ERR": f"Answer the question based on the conversation between a human and AI.\nQuestion: Is the AI able to recover from errors that it makes? (a) Yes. (b) No.\nConversation:\nUser: {{src}}\nAI: {{pred}}\nAnswer:",
            },
        }

        self.tasks = self.templates.keys()
        self.aspects = list(
            {aspect for task in self.tasks for aspect in self.templates[task]}
        )

        if not model:
            self.model_path = hf_hub_download(
                repo_id=model_name_or_path, filename=model_basename
            )

            self.lcpp_llm = Llama(
                model_path=self.model_path,
                n_threads=2,  # CPU cores
                logits_all=True,
            )
        else:
            self.lcpp_llm = model

    def add_template(self, task, code, prompt):
        """
        This method adds a template to the list of pre-defined template.
        Please try to follow the following example pattern to ensure consistency.
        Example:
        "diag": {
            "COH": f"Answer the question based on the conversation between a human and AI.
            \nQuestion: Is the AI coherent and maintains a good conversation flow throughout the conversation?
            (a) Yes. (b) No.\nConversation:\nUser: {{src}}\nAI: {{pred}}\nAnswer:",
        }

        Args:
            task (str): Task name.
            code (str): Aspect code.
            prompt (str): Aspect prompt.
        """
        assert isinstance(task, str), "task must be a string."
        assert isinstance(code, str), "code must be a string."
        assert isinstance(prompt, str), "prompt must be a string."

        self.templates[task][code] = prompt

    def get_prompt(self, aspect, task, src, pred, custom_prompt):
        """
        This method returns a prompt template given a task description, and an aspect to evaluate.
        Args:
            src (str): Source text.
            pred (str): Candidate sentence.
            aspect (str): Aspect to evaluate.
            task (str): Task description.
            custom_prompt (dict): Custom prompt template. Defaults to None.
                Must contain the following keys: "task", "aspect".
        Returns:
            str: Prompt template.
        """
        if aspect and task:
            assert (
                aspect in self.templates[task]
            ), f"Aspect {aspect} is not available for task {task}."
            assert self.templates[task][
                aspect
            ], f"Prompt template for aspect {aspect} and task {task} is non-existent. Please specify a prompt template."

        template = (
            self.templates[task][aspect]
            if (aspect and task)
            else str(custom_prompt["task"])
            + "\nQuestion: "
            + str(custom_prompt["aspect"])
            + "(a) Yes. (b) No.\nConversation:\nUser: {src}\nAI: {pred}\nAnswer:"
        )

        # Replace placeholders with source and candidate sentence
        template = template.replace("{src}", src)
        template = template.replace("{pred}", pred)

        return template

    def compute(self, source, pred, prompt=None, aspect=None, task=None):
        """
        This method computes the GPTScore for a candidate sentence given a source text,
        a prompt template, an aspect to evaluate, and a task description.
        Args:
            source (str): Source text.
            pred (str): Candidate sentence.
            prompt (dict, optional): Custom prompt template. Defaults to None.
                Must contain the following keys: "task", "aspect", "name".
            aspect (str, optional): Aspect to evaluate. Defaults to None.
            task (str, optional): Task description. Defaults to None.
        Returns:
            score (float): Score for the candidate sentence.
        """
        assert isinstance(source, str), "Source must be a string."
        assert isinstance(pred, str), "Pred must be a string."

        # If prompt is given, check that it is a string
        if prompt:
            assert isinstance(prompt, dict), "prompt must be a dict."
            assert not aspect, "aspect must not be given if prompt is given."
            assert not task, "aspect must not be given if prompt is given."
            assert (
                "task" in prompt.keys() and "aspect" in prompt.keys()
            ), "prompt must contain the following keys: 'task', 'aspect'"
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
        prompt = self.get_prompt(aspect, task, source, pred, prompt)

        response = self.lcpp_llm.create_completion(
            prompt=prompt,
            max_tokens=500,
            temperature=0.5,
            top_p=0.95,
            logprobs=1,
            repeat_penalty=1.2,
            top_k=50,
            echo=True,
        )

        # Compute logprobs
        # Find the end position of the input...
        i = response["choices"][0]["logprobs"]["text_offset"].index(len(prompt))
        if i == 0:
            i = i + 1

        # Get logprobs
        loss = -sum(
            response["choices"][0]["logprobs"]["token_logprobs"][i:-1]
        )  # ignore the last '.'
        avg_loss = loss / (
            len(response["choices"][0]["logprobs"]["text_offset"]) - i - 1
        )  # 1 is the last '.'

        return avg_loss
