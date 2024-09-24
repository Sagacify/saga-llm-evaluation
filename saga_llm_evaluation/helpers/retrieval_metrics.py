from langchain.evaluation import load_evaluator
from llama_index.core.evaluation import RetrieverEvaluator


class Relevance:
    def __init__(self, llm=None) -> None:
        """
        Relevance computes the relevance score of retrieved information relatively to the query.
        In other words, it validates that the retrieved information is related to the query topic.

        Args:
            llm (LangChain BaseLanguageModel): model used for evaluation. If None,\
                the model is chosen as "gpt-4" by default.
        """

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
            dict: Relevance score for the candidate sentence. The dictionary contains the following keys:

                - score (int): Relevance score. Binary integer value (0 or 1), where 1 indicates that the sentence is\
                    relevant and 0 indicates that the sentence is irrelevant.
                - value (str): Relevance value. Y or N, where Y indicates that the sentence is relevant and N\
                    indicates that the sentence is irrelevant.
                - reasoning (str): Reasoning for the relevance score.
        """

        result = self.evaluator.evaluate_strings(prediction=contexts, input=query)

        return result


class Accuracy:
    def __init__(self, index, k=2) -> None:
        """
        Accuracy computes the accuracy of the retrieved information relatively to the query.
        In other words, it validates that the retrieved information is correct.

        Args:
            index (llama_index.core.index.Index): Index containing the information to retrieve.
            k (int, optional): Number of similar items to retrieve. Defaults to 2.
        """
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
            dict: Dictionary containing the following metrics:

                - mrr (float): Mean Reciprocal Rank (MRR) metric.
                - hit_rate (float): Hit Rate metric.
                - precision (float): Precision metric.
                - recall (float): Recall metric.
        """

        eval_result = self.retriever_evaluator.evaluate(
            query=query, expected_ids=expected_ids
        )

        return eval_result.metric_vals_dict
