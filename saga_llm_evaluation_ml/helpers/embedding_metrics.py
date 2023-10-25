from evaluate import load


class BERTScore:
    def __init__(self, lan="en", model_type=None):
        """
        BERTScore computes a similarity score for each token in the candidate sentence with each
        token in the reference sentence.
        The final score is the average of the similarity scores of all tokens in the candidate sentence.

        Args:
            lan (str, optional): language to use. Defaults to "en", It may also be "fr". Depending
            on the language, a different model is used by default.
            model_type (sr, optional): Model to use. Defaults to None. If None, a default model is
            used depending on the language (see above).
        """
        if lan == "fr":
            self.model_type = (
                "distilbert-base-multilingual-cased" if not model_type else model_type
            )  # TODO; find uncased version
        elif lan == "en":
            self.model_type = (
                "distilbert-base-uncased" if not model_type else model_type
            )
        self.metric = load("bertscore")

    def compute(self, references, predictions, **kwargs):
        """
        Args:
            references (list): List of reference sentences.
            predictions (list): List of candidate sentences.

        Returns:
            list: List of scores for each candidate sentence. Contains a list of scores for
            precisions, recalls, and F1 scores.
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
    def __init__(self, featurize_model_name="gpt2"):
        """
        MAUVE score computes the difference between the candidate sentence distribution
        and the reference sentence distribution.
        The bigger the MAUVE score, the better.
        """
        self.metric = load("mauve")
        self.featurize_model_name = featurize_model_name

    def compute(self, references, predictions, **kwargs):
        """
        Args:
            references (list): List of reference sentences.
            predictions (list): List of candidate sentences.

        Returns:
            list: List of MAUVE scores for each candidate sentence.
        """
        return self.metric.compute(
            predictions=predictions,
            references=references,
            featurize_model_name=self.featurize_model_name,
            **kwargs
        )
