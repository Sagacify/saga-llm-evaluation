Module saga_llm_evaluation_ml.helpers.llm_metrics
=================================================

Classes
-------

`GEval(model=False, model_name_or_path='TheBloke/Llama-2-7b-Chat-GGUF', model_basename='llama-2-7b-chat.Q4_K_M.gguf')`
:   This class implements the GEval evaluation metric for generative language models.
    It is inspired by the GEval metric proposed in https://arxiv.org/pdf/2303.16634.pdf.
    Args:
        model (Llama model): model used for evaluation. If False, the model is downloaded from the HuggingFace Hub.
        model_name_or_path (str): Model name or path. Defaults to "TheBloke/Llama-2-7b-Chat-GGUF".
        model_basename (str): Model basename. Defaults to "llama-2-7b-chat.Q4_K_M.gguf".

    ### Methods

    `add_aspect(self, code, name, prompt)`
    :   This method adds an aspect to the list of pre-defined aspects.
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

    `add_task(self, name, definition)`
    :   This method adds a task to the list of pre-defined tasks.
        Please try to follow the following example pattern to ensure consistency.
        Example:
        "summ": "You will be given one summary written for a news article. Your task is to rate the summary on one metric. Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.",
        
        Args:
            name (str): Task name.
            definition (str): Task description.

    `compute(self, source, pred, task=None, aspect=None, custom_prompt=None)`
    :   This method computes the GEval score for a candidate sentence given a source text,
        a prompt template, an aspect to evaluate, and a task description.
        Args:
            source (str): Source text.
            pred (str): Candidate sentence to evaluate.
            task (str, optional): Definition of the task.
            aspect (str, optional): Evaluation criterion code.
            custom_prompt (dict, optional): Custom prompt template. Defaults to None.
        
        Returns:
            score (float): Score for the candidate sentence.

    `get_cot(self, prompt)`
    :   This method returns a chain of thoughts given a prompt template.
        Args:
            prompt (str): Prompt template.
        
        Returns:
            cot (str): Chain of thoughts.

    `get_prediction(self, prompt)`
    :   This method returns a prediction given a prompt template.
        Args:
            prompt (str): Prompt template.
        
        Returns:
            response (dict): Response from the model.

    `get_prompt(self, src, pred, task, aspect, custom_prompt)`
    :   Args:
            src (str): Source text.
            pred (str): Candidate sentence to evaluate.
            task (str): Definition of the task.
            aspect (str): Evaluation criterion code.
            custom_prompt (dict): Custom prompt template.
                Must contain the following keys: "task", "aspect", "name".

    `get_score(self, prompt)`
    :   Args:
            prompt (str): Prompt template.
        
        Returns:
            score (float): Score for the candidate sentence.

`GPTScore(model=False, model_name_or_path='TheBloke/Llama-2-7b-Chat-GGUF', model_basename='llama-2-7b-chat.Q4_K_M.gguf')`
:   This class implements the GPTScore evaluation metric for generative language models.
    It is inspired by the GPTScore metric proposed in https://arxiv.org/pdf/2302.04166.pdf.
    Args:
        model (Llama model): model used for evaluation. If False, the model is downloaded from the HuggingFace Hub.
        model_name_or_path (str): Model name or path. Defaults to "TheBloke/Llama-2-7b-Chat-GGUF".
        model_basename (str): Model basename. Defaults to "llama-2-7b-chat.Q4_K_M.gguf".

    ### Methods

    `add_template(self, task, code, prompt)`
    :   This method adds a template to the list of pre-defined template.
                Please try to follow the following example pattern to ensure consistency.
                Example:
                "diag": {
                    "COH": f"Answer the question based on the conversation between a human and AI.
                    
        Question: Is the AI coherent and maintains a good conversation flow throughout the conversation?
                    (a) Yes. (b) No.
        Conversation:
        User: {{src}}
        AI: {{pred}}
        Answer:",
                }
        
                Args:
                    task (str): Task name.
                    code (str): Aspect code.
                    prompt (str): Aspect prompt.

    `compute(self, source, pred, prompt=None, aspect=None, task=None)`
    :   This method computes the GPTScore for a candidate sentence given a source text,
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

    `get_prompt(self, aspect, task, src, pred, custom_prompt)`
    :   This method returns a prompt template given a task description, and an aspect to evaluate.
        Args:
            src (str): Source text.
            pred (str): Candidate sentence.
            aspect (str): Aspect to evaluate.
            task (str): Task description.
            custom_prompt (dict): Custom prompt template. Defaults to None.
                Must contain the following keys: "task", "aspect".
        Returns:
            str: Prompt template.

`SelfCheckGPT(model, eval_model=False, eval_model_name_or_path='TheBloke/Llama-2-7b-Chat-GGUF', eval_model_basename='llama-2-7b-chat.Q4_K_M.gguf')`
:   This class implements the self-check GPT evaluation metric for generative language models.
    It is inspired by the self-check metric proposed in https://arxiv.org/pdf/2303.08896.pdf.
    Args:
        model (transformers.PreTrainedModel): GPT model to evaluate.
        eval_model (LLama model, optional): Evaluation model. If False, the evaluation model is
        downloaded from the HuggingFace Hub.
        eval_model_name_or_path (str): Evaluation model name or path. Defaults to "TheBloke/Llama-2-7b-Chat-GGUF".
        eval_model_basename (str): Evaluation model basename. Defaults to "llama-2-7b-chat.Q4_K_M.gguf".

    ### Methods

    `compute(self, question, pred, n_samples)`
    :   Args:
            question (str): Question asked to the model for which it generated $pred.
            pred (str): Candidate sentence.
            n_samples (int): Number of samples to generate.
        
        Returns:
            score (float): Score for the candidate sentence.

    `get_prompt(self, pred, sample, question)`
    :   This method returns a prompt template given a candidate sentence, a sample sentence, and a question.
        Args:
            pred (str): Candidate sentence.
            sample (str): Sample sentence.
            question (str): Question asked to the model for which it generated $pred.
        
        Returns:
            str: Prompt template.

    `get_prompts(self, pred, samples, question)`
    :   This method returns a list of prompt templates given a candidate sentence, a list
        of sample sentences, and a question.
        Args:
            pred (str): Candidate sentence.
            samples (list of str): List of sample sentences.
            question (str): Question asked to the model for which it generated $pred.
        
        Returns:
            list: List of prompt templates.