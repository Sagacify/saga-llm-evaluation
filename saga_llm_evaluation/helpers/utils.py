import json
import re
import nltk
import string
from collections import Counter

import torch
from elemeta.nlp.extractors.high_level.regex_match_count import RegexMatchCount
from elemeta.nlp.extractors.high_level.word_regex_matches_count import (
    WordRegexMatchesCount,
)
from elemeta.nlp.extractors.low_level.abstract_metafeature_extractor import (
    AbstractMetafeatureExtractor,
)
from elemeta.nlp.metafeature_extractors_runner import MetafeatureExtractorsRunner
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from langchain_community.chat_models import ChatLlamaCpp

from langchain_openai import ChatOpenAI

nltk.download("punkt_tab")

NO_ANS = "[CLS]"
INVALID_QUESTION = -1

# pylint:disable=too-many-boolean-expressions


def load_json(path):
    """
    Load a json file from a given path.

    Args:
        path (str) : path to the json file
    Returns:
        dict: dictionary of the json file
    """
    with open(path) as json_file:
        o_file = json_file.read()
    return json.loads(o_file)


def filter_questions(exp_ans, pred_ans):
    """
    Check if the expected answer and the predicted answer are the same.

    Args:
        exp_ans (str) : expected answer
        pred_ans (str) : predicted answer
    Returns:
        str: "VALID" if the answers are the same, "NO MATCH" otherwise
    """
    if pred_ans == NO_ANS:
        return "NO MATCH"
    if clean_text(exp_ans) != clean_text(pred_ans):
        return "NO MATCH"
    return "VALID"


def clean_text(text):
    """
    Clean a text by removing punctuation and (some) stopwords.

    Args:
        text (str) : text to clean
    Returns:
        str: cleaned text
    """
    # TODO: improve
    # TODO: add support to french language
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the|in|our)\b", " ", text)
    return re.sub(" +", " ", text).strip()


def raw_f1_score(a_gold, a_pred):
    """
    Compute the raw F1 score between two answers.

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


def non_personal(question, nlp, lan="en"):
    """
    Check if a question contains personal pronouns.

    Args:
        question (str) : question to check
        nlp (spacy.lang) : spacy language model
        lan (str) : language of the question. Defaults to "en".

    Returns:
        bool: True if the question does not contain personal pronouns, False otherwise
    """
    question_tok = nlp(question)
    for tok in question_tok:
        if tok.dep_ == "nsubj" and lan == "en":
            if (
                tok.text.lower() == "i" or tok.text.lower() == "you"
            ):  # TODO: add support to french language
                return False
        elif tok.dep_ == "poss" and lan == "en":
            if (
                tok.text.lower() == "my" or tok.text.lower() == "your"
            ):  # TODO: add support to french language
                return False
        # french
        elif tok.dep_ == "nsubj" and lan == "fr":
            if (
                tok.text.lower() == "je"
                or tok.text.lower() == "tu"
                or tok.text.lower() == "vous"
            ):
                return False
        elif tok.dep_ == "poss" and lan == "fr":
            if tok.text.lower() in [
                "mon",
                "ton",
                "votre",
                "ma",
                "ta",
                "vos",
                "mes",
                "tes",
            ]:
                return False
    return True


def get_llama_model(
    repo_id: str = "TheBloke/Llama-2-7b-Chat-GGUF",
    filename: str = "llama-2-7b-chat.Q2_K.gguf",
    model_path=False,
):
    """
    Download and return a Llama model from HuggingFace Hub.

    Args:
        repo_id (str) : HuggingFace Hub repo id of the model. Defaults to "TheBloke/Llama-2-7b-Chat-GGUF".
        filename (str) : model filename to download. Defaults to "llama-2-7b-chat.Q2_K.gguf".
        model_path (str) : path to the model locally to avoid downloading. Defaults to False.

    Returns:
        llama_cpp.Llama: Llama model
    """
    if not model_path:
        model_path = hf_hub_download(repo_id, filename)

    if torch.cuda.is_available():
        lcpp_llm = Llama(
            model_path=model_path,
            main_gpu=0,
            n_gpu_layers=40,  # check this
            n_batch=1024,
            logits_all=True,
            n_ctx=1024,
            device="cuda",
        )
    else:
        lcpp_llm = Llama(
            model_path=model_path,
            logits_all=True,
            n_ctx=1024,
        )

    return lcpp_llm


def get_langchain_llama_model(
    repo_id: str = "TheBloke/Llama-2-7b-Chat-GGUF",
    filename: str = "llama-2-7b-chat.Q2_K.gguf",
    model_path=False,
):
    """
    Download and return a LlamaCPP model from LangChain, loaded from the HuggingFace Hub.

    Args:
        repo_id (str) : HuggingFace Hub repo id of the model. Defaults to "TheBloke/Llama-2-7b-Chat-GGUF".
        filename (str) : model filename to download. Defaults to "llama-2-7b-chat.Q2_K.gguf".
        model_path (str) : path to the model locally to avoid downloading. Defaults to False.

    Returns:
        langchain_community.chat_models.ChatLlamaCpp: LlamaCPP model from LangChain.
    """
    if not model_path:
        model_path = hf_hub_download(repo_id, filename)

    if torch.cuda.is_available():
        lcpp_llm = ChatLlamaCpp(
            model_path=model_path,
            n_gpu_layers=40,  # check this
            n_batch=1024,
            logits_all=True,
            logprobs=1,
            n_ctx=1024,
            device="cuda",
        )
    else:
        lcpp_llm = ChatLlamaCpp(
            model_path=model_path,
            logits_all=True,
            logprobs=1,
            n_ctx=1024,
        )
    return lcpp_llm


def get_langchain_gpt_model(version="gpt-3.5-turbo-0125"):
    """
    Return a GPT model from Langchain OpenAI.

    Args:
        version (str) : model version

    Returns:
        langchain_openai.ChatOpenAI: GPT model from LangChain OpenAI.
    """
    return ChatOpenAI(model=version)


def filter_class_input(args, python_function: object, drop=None):
    """
    Filters input arguments for a given class.

    Args:
        args (dict): dictionary of arguments
        python_class (object): class to filter arguments for
        drop (list, optional): list of arguments to drop. Defaults to None.
    Returns:
        dict: filtered dictionary of arguments
    """
    init_varnames = set(python_function.__code__.co_varnames)
    if drop:
        args = {k: v for k, v in args.items() if k not in init_varnames}
    args = {k: v for k, v in args.items() if k in init_varnames}
    return args


def check_list_type(array: list, list_type: type):
    """
    Check if an array is a list of a given type.

    Args:
        array (list): array to check
        list_type (type): type to check

    Returns:
        bool: True if the array is a list of the given type, False otherwise
    """
    if not isinstance(array, list):
        return False
    return all(isinstance(item, list_type) for item in array)


# pylint:disable=invalid-name
class MetadataExtractor:
    def __init__(self):
        """
        Initializes the metadata extractor.
        """
        self.metadata_extractor = MetafeatureExtractorsRunner()

    def add_word_regex_matches_count(self, regex_rule, name=None):
        """
        Adds a regex rule to the metadata extractor.
        For a given regex return the number of words matching the regex.

        Args:
            regex_rule (str): regex rule to add
        """
        self.metadata_extractor.add_metafeature_extractor(
            WordRegexMatchesCount(regex=regex_rule, name=name)
        )

    def add_regex_match_count(self, regex_rule, name=None):
        """
        Adds a regex rule to the metadata extractor.
        For a given regex return the number of matches it has in the text.

        Args:
            regex_rule (str): regex rule to add
        """
        self.metadata_extractor.add_metafeature_extractor(
            RegexMatchCount(regex=regex_rule, name=name)
        )

    def add_custom_extractor(self, extractor: AbstractMetafeatureExtractor):
        """
        Adds a custom extractor to the metadata extractor.

        Args:
            extractor (object): extractor to add
        """
        self.metadata_extractor.add_metafeature_extractor(extractor)

    def compute(self, text):
        """
        Computes metadata from a text using elemeta library and returns a dictionary of metadata.

        Args:
            text (str): text to extract metadata from

        Returns:
            dict: dictionary of metadata
        """
        return self.metadata_extractor.run(text)
