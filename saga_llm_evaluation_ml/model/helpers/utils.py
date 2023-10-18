import json
from elemeta.nlp.metafeature_extractors_runner import (
    MetafeatureExtractorsRunner,
)
from elemeta.nlp.extractors.high_level.word_regex_matches_count import (
    WordRegexMatchesCount,
)
from elemeta.nlp.extractors.high_level.regex_match_count import RegexMatchCount


def load_json(path):
    with open(path) as json_file:
        o_file = json_file.read()
    return json.loads(o_file)


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
