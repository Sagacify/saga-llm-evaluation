from langchain.evaluation import load_evaluator
from llama_index.core.evaluation import RetrieverEvaluator


class Relevance:
    def __init__(self, llm=None) -> None:
        self.llm = llm if llm else None
        self.criterion = {
            "retrieval_relevance": "Are the retrieved information related to the query?",
        }
        self.evaluator = load_evaluator(
            "criteria", criteria=self.criterion, llm=self.llm
        )

    def compute(self, contexts: list, query: str):
        """
        This method computes the relevance score of retrieved information relatively to the query.
        In other words, it validates that the retrieved information is related to the query topic.
        Args:
            contexts (list): List of retrieved information.
            query (str): Query topic.
        Returns:
            result (dict): Relevance score for the candidate sentence. The dictionary contains the following keys:
                - "score": Relevance score. Binary integer value (0 or 1), where 1 indicates that the sentence is
                    relevant and 0 indicates that the sentence is irrelevant.
                - "value": Relevance value. Y or N, where Y indicates that the sentence is relevant and N
                    indicates that the sentence is irrelevant.
                - "reasoning": Reasoning for the relevance score.
        """

        result = self.evaluator.evaluate_strings(prediction=contexts, input=query)

        return result


class Accuracy:
    def __init__(self, index, k=2) -> None:
        self.retriever = index.as_retriever(similarity_top_k=k)
        self.retriever_evaluator = RetrieverEvaluator.from_metric_names(
            ["mrr", "hit_rate", "precision", "recall"], retriever=self.retriever
        )

    def compute(self, query: str, expected_ids: list) -> dict:
        """
        This method computes a series of metrics of the retrieved information relatively to the query.
        Args:
            query (str): Query topic.
            expected_ids (list): List of expected IDs of the retrieved information (nodes of the index).
        Returns:
            result (dict): Dictionary containing the following metrics:
                - "mrr": Mean Reciprocal Rank (MRR) metric.
                - "hit_rate": Hit Rate metric.
                - "precision": Precision metric.
                - "recall": Recall metric.
        """

        eval_result = self.retriever_evaluator.evaluate(
            query=query, expected_ids=expected_ids
        )

        return eval_result.metric_vals_dict
