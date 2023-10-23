import spacy
import torch
from evaluate import load
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelWithLMHead,
    AutoTokenizer,
)

from saga_llm_evaluation_ml.model.helpers.embedding_metrics import BERTScore
from saga_llm_evaluation_ml.model.helpers.utils import (
    INVALID_QUESTION,
    NO_ANS,
    filter_questions,
    non_personal,
)


# pylint:disable=too-many-locals
class BLEURTScore:
    def __init__(self, checkpoint="BLEURT-tiny"):
        """
        BLEURT is a learnt metric that uses BERT to compute a similarity score for each token
        in the candidate sentence with each token in the reference sentence.

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


class QSquared:
    def __init__(self, lan="en") -> None:
        """
        Q² is a reference-free metric that aims to evaluate the factual consistency of knowledge-grounded
        dialogue systems. The approach is based on automatic question generation and question answering
        Source: https://github.com/orhonovich/q-squared

        Args:
            lan (str, optional): Language to use. Defaults to "en", It may also be "fr".
        """
        self.qa_tokenizer = AutoTokenizer.from_pretrained(
            "ktrapeznikov/albert-xlarge-v2-squad-v2"
        )
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(
            "ktrapeznikov/albert-xlarge-v2-squad-v2"
        )
        self.qg_tokenizer = AutoTokenizer.from_pretrained(
            "mrm8488/t5-base-finetuned-question-generation-ap"
        )
        self.qg_model = AutoModelWithLMHead.from_pretrained(
            "mrm8488/t5-base-finetuned-question-generation-ap"
        )
        assert lan in ["fr", "en"], "Language must be either fr or en"
        self.bert_score = BERTScore(lan=lan)

        if lan == "fr":
            self.nlp = spacy.load("fr_core_news_sm")
        elif lan == "en":
            self.nlp = spacy.load("en_core_web_sm")

    def get_answer(
        self, question: str, text: str
    ):  # Code taken from https://huggingface.co/transformers/task_summary.html
        """
        Search for the answer in the text given the question.
        Args:
            question (str) : question to ask
            text (str) : text to search in
        Returns:
            answer (str) : answer to the question
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
        Args:
            text (str) : text to search in
        Returns:
            candidates (str) : candidates answers
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
        self, answer, context, max_length=128, beam_size=5, num_return=5
    ):
        """
        Get the n best questions for a given answer, given the context. "Beam" is the name of the
        approach
        Args:
            answer (str) : answer to the question
            context (str) : context to search in
            max_length (int, optional) : max length of the generated question. Defaults to 128.
            beam_size (int, optional) : beam size. Defaults to 5.
            num_return (int, optional) : number of questions to return. Defaults to 5.
        Returns:
            all_questions (list) : n best questions
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

    def single_question_score(self, question, answer, response, knowledge):
        """
        Given a candidate pair of question and answer (generated from the candidate text), get the
        score of the aswer given by taking as a context the knowledge that the LLM was given.
        The higher the F1-score, the more the model we are trying to evaluate is consistent
        with the knowledge.
        Args:
            question (str) : cadidate question (generated from the candidate text)
            answer (str) : candidate answer (generated from the candidate text)
            response (str) : text generated by the LLM
            knowledge (str) : knowledge given as a context to the LLM

        Returns:
            score, answer (tuple) : bert-score of the knowledge answer, knowledge answer
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

    def compute(self, response, knowledge, single=False, remove_personal=True):
        """
        Compute the Q² score for a given response and knowledge.
        Args:
            response (str) : text generated by the LLM
            knowledge (str) : knowledge given as a context to the LLM
            single (bool) : if True, only one question is generated for each candidate answer.
                            Defaults to False.
            remove_personal (bool) : if True, remove questions that contain personal pronouns.
                                     Defaults to True.
        Returns:
            avg_f1 (float) : average F1-bert-score of the knowledge answers (Q² score)
        """

        f1_bert_score = 0
        num_questions = 0

        # valid_questions = []
        # valid_cands = []
        # knowledge_answers = []
        # scores = []

        candidates = self.get_answer_candidates(response)
        for cand in candidates:
            questions = self.get_questions_beam(cand, response)
            for question in questions:
                if not remove_personal or non_personal(question, self.nlp):
                    question_score, _ = self.single_question_score(
                        question, cand, response, knowledge
                    )
                    if question_score != INVALID_QUESTION:
                        num_questions += 1
                        f1_bert_score += question_score

                        # valid_questions.append(question)
                        # valid_cands.append(cand)
                        # knowledge_answers.append(knowledge_ans)
                        # scores.append(question_score)

                        if single:
                            break

        if num_questions:
            avg_f1 = f1_bert_score / num_questions
        else:
            avg_f1 = INVALID_QUESTION
        return avg_f1  # , valid_questions, valid_cands, knowledge_answers, scores
