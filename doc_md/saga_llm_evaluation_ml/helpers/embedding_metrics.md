Module saga_llm_evaluation_ml.helpers.embedding_metrics
=======================================================

Classes
-------

`BERTScore(lan='en', model_type=None)`
:   BERTScore computes a similarity score for each token in the candidate sentence with each
    token in the reference sentence.
    The final score is the average of the similarity scores of all tokens in the candidate sentence.
    
    Args:
        lan (str, optional): language to use. Defaults to "en", It may also be "fr". Depending
        on the language, a different model is used by default.
        model_type (sr, optional): Model to use. Defaults to None. If None, a default model is
        used depending on the language (see above).

    ### Methods

    `compute(self, references, predictions, **kwargs)`
    :   Args:
            references (list): List of reference sentences.
            predictions (list): List of candidate sentences.
        
        Returns:
            list: List of scores for each candidate sentence. Contains a list of scores for
            precisions, recalls, and F1 scores.

`MAUVE(featurize_model_name='gpt2')`
:   MAUVE score computes the difference between the candidate sentence distribution
    and the reference sentence distribution.
    The bigger the MAUVE score, the better.

    ### Methods

    `compute(self, references, predictions, **kwargs)`
    :   Args:
            references (list): List of reference sentences.
            predictions (list): List of candidate sentences.
        
        Returns:
            list: List of MAUVE scores for each candidate sentence.