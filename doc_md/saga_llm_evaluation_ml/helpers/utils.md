Module saga_llm_evaluation_ml.helpers.utils
===========================================

Functions
---------

    
`clean_text(text)`
:   clean a text by removing punctuation and (some) stopwords.
    Args:
        text (str) : text to clean
    Returns:
        str : cleaned text

    
`filter_questions(exp_ans, pred_ans)`
:   check if the expected answer and the predicted answer are the same.
    Args:
        exp_ans (str) : expected answer
        pred_ans (str) : predicted answer
    Returns:
        str : "VALID" if the answers are the same, "NO MATCH" otherwise

    
`get_llama_model(repo_id: str = 'TheBloke/Llama-2-7b-Chat-GGUF', filename: str = 'llama-2-7b-chat.Q4_K_M.gguf')`
:   Download and return a Llama model from HuggingFace Hub.
    Args:
        repo_id (str) : HuggingFace Hub repo id
        filename (str) : model filename

    
`load_json(path)`
:   

    
`non_personal(question, nlp)`
:   check if a question contains personal pronouns.
    Args:
        question (str) : question to check
        nlp (spacy.lang) : spacy language model
    Returns:
        bool : True if the question does not contain personal pronouns, False otherwise

    
`raw_f1_score(a_gold, a_pred)`
:   compute the raw F1 score between two answers.
    Args:
        a_gold (str) : expected answer
        a_pred (str) : predicted answer
    Returns:
        float : F1 score

Classes
-------

`MetadataExtractor()`
:   

    ### Methods

    `add_regex_match_count(self, regex_rule, name=None)`
    :   Adds a regex rule to the metadata extractor.
        For a given regex return the number of matches it has in the text.
        
        Args:
            regex_rule (str): regex rule to add

    `add_word_regex_matches_count(self, regex_rule, name=None)`
    :   Adds a regex rule to the metadata extractor.
        For a given regex return the number of words matching the regex.
        
        Args:
            regex_rule (str): regex rule to add

    `compute(self, text)`
    :   Computes metadata from a text using elemeta library and returns a dictionary of metadata.
        
        Args:
            text (str): text to extract metadata from
        
        Returns:
            dict: dictionary of metadata