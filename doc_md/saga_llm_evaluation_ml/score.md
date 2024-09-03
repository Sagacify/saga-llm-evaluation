Module saga_llm_evaluation_ml.score
===================================

Classes
-------

`LLMScorer(model=None, lan='en', bleurt_model='BLEURT-tiny', mauve_model='gpt2', selfcheckgpt_eval_model_name_or_path='TheBloke/Llama-2-7b-Chat-GGUF', selfcheckgpt_eval_model_basename='llama-2-7b-chat.Q4_K_M.gguf', geval_model_name_or_path='TheBloke/Llama-2-7b-Chat-GGUF', geval_model_basename='llama-2-7b-chat.Q4_K_M.gguf', gptscore_model_name_or_path='TheBloke/Llama-2-7b-Chat-GGUF', gptscore_model_basename='llama-2-7b-chat.Q4_K_M.gguf')`
:   

    ### Methods

    `add_geval_aspect(self, code, name, prompt)`
    :   This function adds an aspect to the GEval metric.
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

    `add_geval_task(self, name, definition)`
    :   This function adds a task to the GEval metric.
        Please try to follow the following example pattern to ensure consistency.
        Example:
        "summ": "You will be given one summary written for a news article.
            Your task is to rate the summary on one metric.
            Please make sure you read and understand these instructions carefully.
            Please keep this document open while reviewing, and refer to it as needed.",
        Args:
            name (str): Name of the task.
            definition (str): Definition of the task.

    `add_gptscore_template(self, task, code, prompt)`
    :   This function adds a template to the GPTScore metric.
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
                    task (str): Task of the template.
                    code (str): Code of the aspect.
                    prompt (str): Prompt of the aspect.

    `score(self, llm_input: str, prompt: str, prediction: str, context: str = None, reference: str = None, n_samples: int = 5, task: str = None, aspects: list = None, custom_prompt: dict = None)`
    :   Args:
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