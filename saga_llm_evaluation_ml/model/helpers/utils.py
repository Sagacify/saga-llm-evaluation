import json
import re
import string
from collections import Counter

from elemeta.nlp.extractors.high_level.regex_match_count import RegexMatchCount
from elemeta.nlp.extractors.high_level.word_regex_matches_count import (
    WordRegexMatchesCount,
)
from elemeta.nlp.metafeature_extractors_runner import MetafeatureExtractorsRunner

NO_ANS = "[CLS]"
INVALID_QUESTION = -1


def load_json(path):
    with open(path) as json_file:
        o_file = json_file.read()
    return json.loads(o_file)


def filter_questions(exp_ans, pred_ans):
    """
    check if the expected answer and the predicted answer are the same.
    Args:
        exp_ans (str) : expected answer
        pred_ans (str) : predicted answer
    Returns:
        str : "VALID" if the answers are the same, "NO MATCH" otherwise
    """
    if pred_ans == NO_ANS:
        return "NO MATCH"
    if clean_text(exp_ans) != clean_text(pred_ans):
        return "NO MATCH"
    return "VALID"


def clean_text(text):
    """
    clean a text by removing punctuation and (some) stopwords.
    Args:
        text (str) : text to clean
    Returns:
        str : cleaned text
    """
    # TODO: improve
    # TODO: add support to french language
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the|in|our)\b", " ", text)
    return re.sub(" +", " ", text).strip()


def raw_f1_score(a_gold, a_pred):
    """
    compute the raw F1 score between two answers.
    Args:
        a_gold (str) : expected answer
        a_pred (str) : predicted answer
    Returns:
        float : F1 score
    """
    if a_pred == "":
        return 0
    gold_toks = clean_text(a_gold).split()
    pred_toks = clean_text(a_pred).split()
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1_score = (2 * precision * recall) / (precision + recall)
    return f1_score


def non_personal(question, nlp):
    """
    check if a question contains personal pronouns.
    Args:
        question (str) : question to check
        nlp (spacy.lang) : spacy language model
    Returns:
        bool : True if the question does not contain personal pronouns, False otherwise
    """
    question_tok = nlp(question)
    for tok in question_tok:
        if tok.dep_ == "nsubj":
            if (
                tok.text.lower() == "i" or tok.text.lower() == "you"
            ):  # TODO: add support to french language
                return False
        elif tok.dep_ == "poss":
            if (
                tok.text.lower() == "my" or tok.text.lower() == "your"
            ):  # TODO: add support to french language
                return False
    return True


# pylint:disable=invalid-name
class MetadataExtractor:
    def __init__(self):
        self.metadata_extractor = MetafeatureExtractorsRunner()

    def addWordRegexMatchesCount(self, regex_rule, name=None):
        """
        Adds a regex rule to the metadata extractor.
        For a given regex return the number of words matching the regex.

        Args:
            regex_rule (str): regex rule to add
        """
        self.metadata_extractor.add_metafeature_extractor(
            WordRegexMatchesCount(regex=regex_rule, name=name)
        )

    def addRegexMatchCount(self, regex_rule, name=None):
        """
        Adds a regex rule to the metadata extractor.
        For a given regex return the number of matches it has in the text.

        Args:
            regex_rule (str): regex rule to add
        """
        self.metadata_extractor.add_metafeature_extractor(
            RegexMatchCount(regex=regex_rule, name=name)
        )

    def compute(self, text):
        """
        Computes metadata from a text using elemeta library and returns a dictionary of metadata.

        Args:
            text (str): text to extract metadata from

        Returns:
            dict: dictionary of metadata
        """
        return self.metadata_extractor.run(text)
