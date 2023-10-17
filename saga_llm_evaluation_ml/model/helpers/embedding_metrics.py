# TODO: Implement BERTScore
# TODO: Implement MAUVE
from evaluate import load


class BERTScore:
    def __init__(self, model_type="distilbert-base-uncased"):
        """
        BERTScore computes a similarity score for each token in the candidate sentence with each token in the reference sentence.
        The final score is the average of the similarity scores of all tokens in the candidate sentence.

        Args:
            model_type (str, optional): Model type to use. Defaults to "roberta-large".
        """
        self.model_type = model_type
        self.metric = load("bertscore")

    def compute(self, references, predictions, **kwargs):
        """
        Args:
            references (list): List of reference sentences.
            predictions (list): List of candidate sentences.

        Returns:
            list: List of scores for each candidate sentence. Contains a list of scores for precisions, recalls, and F1 scores.
        """
        assert len(references) == len(
            predictions
        ), "Number of references and predictions must be equal."
        assert isinstance(references, list), "References must be a list."
        assert isinstance(predictions, list), "Predictions must be a list."

        return self.metric.compute(
            predictions=predictions,
            references=references,
            model_type=self.model_type,
            **kwargs
        )


class MAUVE:
    def __init__(self):
        """
        MAUVE score computes the difference between the candidate sentence distribution and the reference sentence distribution.
        The bigger the MAUVE score, the better.
        """
        self.metric = load("mauve")

    def compute(self, references, predictions, **kwargs):
        """
        Args:
            references (list): List of reference sentences.
            predictions (list): List of candidate sentences.

        Returns:
            list: List of MAUVE scores for each candidate sentence.
        """
        return self.metric.compute(
            predictions=predictions, references=references, featurize_model_name="gpt2", **kwargs
        )
