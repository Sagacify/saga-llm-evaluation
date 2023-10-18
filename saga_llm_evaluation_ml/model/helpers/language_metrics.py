from evaluate import load


class BLEURTScore:
    def __init__(self, checkpoint="BLEURT-tiny"):
        """
        BLEURT is a learnt metric that uses BERT to compute a similarity score for each token in the candidate sentence with each token in the reference sentence.

        Args:
            checkpoint (str, optional): Checkpoint to use. Defaults to BLEURT-tiny if not specified.
        """
        self.checkpoint = checkpoint
        self.metric = load("bleurt", module_type="metric", checkpoint=self.checkpoint)

    def compute(self, references, predictions, **kwargs):
        """
        Args:
            references (list): List of reference sentences.
            predictions (list): List of candidate sentences.

        Returns:
            list: List of scores for each candidate sentence.
        """
        assert len(references) == len(
            predictions
        ), "Number of references and predictions must be equal."
        assert isinstance(references, list), "References must be a list."
        assert isinstance(predictions, list), "Predictions must be a list."

        return self.metric.compute(
            predictions=predictions, references=references, **kwargs
        )
