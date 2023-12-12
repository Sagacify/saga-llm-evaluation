import spacy
import torch
from evaluate import load
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelWithLMHead,
    AutoTokenizer,
)

from sagacify_llm_evaluation.helpers.embedding_metrics import BERTScore
from sagacify_llm_evaluation.helpers.utils import (
    INVALID_QUESTION,
    NO_ANS,
    check_list_type,
    filter_questions,
    non_personal,
)

# pylint:disable=too-many-locals
# pylint:disable=too-many-nested-blocks


class BLEURTScore:
    def __init__(self, checkpoint="BLEURT-tiny"):
        """
        BLEURT is a learnt metric that uses BERT to compute a similarity score for each token
        in the candidate sentence with each token in the reference sentence.

        :param checkpoint: Checkpoint to use. Defaults to BLEURT-tiny if not specified. \
            Check https://huggingface.co/spaces/evaluate-metric/bleurt for more checkpoints.
        :type checkpoint: str, optional
        """
        self.checkpoint = checkpoint
        self.metric = load("bleurt", module_type="metric", checkpoint=self.checkpoint)

    def compute(self, references, predictions, **kwargs):
        """
        :param references: List of reference sentences.
        :type references: list
        :param predictions: List of candidate sentences.
        :type predictions: list
        :return: List of scores for each candidate sentence.
        :rtype: list
        """
        assert len(references) == len(
            predictions
        ), "Number of references and predictions must be equal."
        assert check_list_type(references, str), "References must be a list of strings."
        assert check_list_type(
            predictions, str
        ), "Predictions must be a list of strings."

        return self.metric.compute(
            predictions=predictions, references=references, **kwargs
        )


class QSquared:
    def __init__(
        self,
        qa_model: str = "ktrapeznikov/albert-xlarge-v2-squad-v2",
        qg_model: str = "mrm8488/t5-base-finetuned-question-generation-ap",
        lang="en",
    ) -> None:
        """
        Q² is a reference-free metric that aims to evaluate the factual consistency of knowledge-grounded
        dialogue systems. The approach is based on automatic question generation and question answering.
        Source: https://github.com/orhonovich/q-squared

        :param qa_model: Huggingface question answering model to use.
        :type qa_model: str
        :param qg_model: Huggingface question generation model to use.
        :type qg_model: str
        :param lan: Language to use. Defaults to "en", It may also be "fr".
        :type lan: str, optional
        """
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_model)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model)
        self.qg_tokenizer = AutoTokenizer.from_pretrained(qg_model)
        self.qg_model = AutoModelWithLMHead.from_pretrained(qg_model)
        assert lang in [
            "fr",
            "en",
        ], "Languages supported are either fr (French) or en (English)"
        self.bert_score = BERTScore(lang=lang)

        if lang == "fr":
            self.nlp = spacy.load("fr_core_news_sm")
        elif lang == "en":
            self.nlp = spacy.load("en_core_web_sm")
        self.lang = lang

    def get_answer(
        self, question: str, text: str
    ):  # Code taken from https://huggingface.co/transformers/task_summary.html
        """
        Search for the answer in the text given the question.

        :param question: Question to ask.
        :type question: str
        :param text: Text to search in.
        :type text: str
        :return: Answer to the question.
        :rtype: str
        """
        inputs = self.qa_tokenizer.encode_plus(
            question, text, add_special_tokens=True, return_tensors="pt"
        )
        input_ids = inputs["input_ids"].tolist()[0]

        answer_start_scores, answer_end_scores = self.qa_model(
            **inputs, return_dict=False
        )

        answer_start = torch.argmax(
            answer_start_scores
        )  # Get the most likely beginning of answer with the argmax of the score
        answer_end = (
            torch.argmax(answer_end_scores) + 1
        )  # Get the most likely end of answer with the argmax of the score

        ans = self.qa_tokenizer.convert_tokens_to_string(
            self.qa_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
        )
        return ans

    def get_answer_candidates(self, text: str):
        """
        Look for candidate aswers that could be answered by the text.

        :param text: Text to search in.
        :type text: str
        :return: Candidates answers.
        :rtype: str
        """
        doc = self.nlp(text)
        candidates = [ent.text for ent in list(doc.ents)]
        noun_chunks = list(doc.noun_chunks)
        for chunk in noun_chunks:
            found = False
            for cand in candidates:
                if chunk.text.lower() == cand.lower():
                    found = True
            if not found:
                candidates.append(chunk.text)
        # candidates += [chunk.text for chunk in list(doc.noun_chunks) if chunk.text not in candidates]
        candidates = [cand for cand in candidates if cand.lower() != "i"]
        return candidates

    def get_questions_beam(
        self,
        answer: str,
        context: str,
        max_length: int = 128,
        beam_size: int = 5,
        num_return: int = 5,
    ):
        """
        Get the n best questions for a given answer, given the context. "Beam" is the name of the approach.

        :param answer: Answer to the question.
        :type answer: str
        :param context: Context to search in.
        :type context: str
        :param max_length: Max length of the generated question. Defaults to 128.
        :type max_length: int, optional
        :param beam_size: Beam size. Defaults to 5.
        :type beam_size: int, optional
        :param num_return: Number of questions to return. Defaults to 5.
        :type num_return: int, optional
        :return: N best questions.
        :rtype: list
        """
        all_questions = []
        input_text = f"answer: {answer}  context: {context} </s>"
        features = self.qg_tokenizer([input_text], return_tensors="pt")

        beam_outputs = self.qg_model.generate(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            max_length=max_length,
            num_beams=beam_size,
            no_repeat_ngram_size=3,
            num_return_sequences=num_return,
            early_stopping=True,
        )

        for beam_output in beam_outputs:
            all_questions.append(
                self.qg_tokenizer.decode(beam_output, skip_special_tokens=True).replace(
                    "question: ", "", 1
                )
            )

        return all_questions

    def single_question_score(
        self, question: str, answer: str, response: str, knowledge: str
    ):
        """
        Given a candidate pair of question and answer (generated from the candidate text), get the
        score of the answer given by taking as a context the knowledge that the LLM was given.
        The higher the F1-score, the more the model we are trying to evaluate is consistent
        with the knowledge.

        :param question: The candidate question (generated from the candidate text).
        :type question: str
        :param answer: The candidate answer (generated from the candidate text).
        :type answer: str
        :param response: The text generated by the LLM.
        :type response: str
        :param knowledge: The knowledge given as a context to the LLM.
        :type knowledge: str

        :return: A tuple containing the BERT-score of the knowledge answer and the knowledge answer itself.
        :rtype: tuple
        """
        pred_ans = self.get_answer(question, response)

        if (
            filter_questions(answer, pred_ans) == "VALID"
        ):  # check if the answer is valid
            knowledge_ans = self.get_answer(question, knowledge)
            if knowledge_ans != NO_ANS:
                score = self.bert_score.compute(
                    references=[answer], predictions=[knowledge_ans]
                )
                return score["f1"][0], knowledge_ans
            return 0, NO_ANS
        return INVALID_QUESTION, INVALID_QUESTION

    def compute(
        self,
        predictions: list,
        knowledges: list,
        single: bool = False,
        remove_personal: bool = True,
    ):
        """
        Compute the Q² score for a given response and knowledge.

        :param predictions: (list or str) List of candidate text generated by the LLM.
        :param knowledges: (list or str) List of knowledge given as a context to the LLM for each candidate text.
        :param single: (bool) If True, only one question is generated for each candidate answer. Defaults to False.
        :param remove_personal: (bool) If True, remove questions that contain personal pronouns. Defaults to True.

        :return: (dict) Dictionary with the following keys:
            - avg_f1 (float): Average F1-score Q² score among all the questions.
        """
        assert check_list_type(
            predictions, str
        ), "Predictions must be a list of strings."
        assert check_list_type(knowledges, str), "Knowledges must be a list of strings."

        # convert to list if single prediction and/or knowledge
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(knowledges, str):
            knowledges = [knowledges]

        assert len(predictions) == len(
            knowledges
        ), "Number of predictions and knowledges must be equal."

        avg_f1s = []

        for prediction, knowledge in zip(predictions, knowledges):
            # set initial values
            f1_bert_score = 0
            num_questions = 0
            scores = []
            candidates = self.get_answer_candidates(prediction)
            for cand in candidates:
                questions = self.get_questions_beam(cand, prediction)
                for question in questions:
                    if not remove_personal or non_personal(
                        question, self.nlp, self.lang
                    ):
                        question_score, _ = self.single_question_score(
                            question, cand, prediction, knowledge
                        )
                        if question_score != INVALID_QUESTION:
                            num_questions += 1
                            f1_bert_score += question_score
                            scores.append(question_score)

                            if single:
                                break

            if num_questions:
                avg_f1 = f1_bert_score / num_questions
            else:
                avg_f1 = INVALID_QUESTION
            avg_f1s.append(avg_f1)

        return {
            "avg_f1": avg_f1s,
        }
