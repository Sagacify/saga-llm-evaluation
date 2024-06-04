import unittest

from saga_llm_evaluation.helpers.utils import MetadataExtractor


class TestMetadataExtractor(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extractor = MetadataExtractor()

    def test_extract_metadata(self):
        """Tests that the MetadataExtractor class extracts the correct metadata."""
        text = "The cat sat on the mat."
        metadata = self.extractor.compute(text)

        # Test a few metadata values
        self.assertEqual(metadata["text_length"], 23)
        self.assertEqual(metadata["unique_word_ratio"], 1)
        self.assertEqual(metadata["unique_word_count"], 6)
        self.assertEqual(metadata["emoji_count"], 0)
        self.assertEqual(metadata["number_count"], 0)
        self.assertEqual(metadata["sentence_count"], 1)
        self.assertEqual(metadata["stop_words_count"], 3)
        self.assertEqual(metadata["syllable_count"], 6)
        self.assertEqual(metadata["sentence_avg_length"], 23)

    def test_add_regex(self):
        """Tests that the MetadataExtractor class extracts the correct metadata when regex rules are added."""
        text = "The cat sat on the mat."
        self.extractor.add_word_regex_matches_count("the")
        self.extractor.add_regex_match_count("the")
        metadata = self.extractor.compute(text)

        # Test a few metadata values
        self.assertEqual(metadata["text_length"], 23)
        self.assertEqual(metadata["unique_word_ratio"], 1)
        self.assertEqual(metadata["unique_word_count"], 6)
        self.assertEqual(metadata["emoji_count"], 0)
        self.assertEqual(metadata["number_count"], 0)
        self.assertEqual(metadata["sentence_count"], 1)
        self.assertEqual(metadata["stop_words_count"], 3)
        self.assertEqual(metadata["syllable_count"], 6)
        self.assertEqual(metadata["sentence_avg_length"], 23)
        self.assertEqual(metadata["word_regex_matches_count"], 1)
        self.assertEqual(metadata["regex_match_count"], 1)

        len_metadata = len(metadata)

        # Check that the metadata is longer when multiple regex rules are added
        self.extractor.add_word_regex_matches_count(
            "cat", name="word_regex_matches_count_cat"
        )
        self.extractor.add_regex_match_count("cat", name="regex_match_count_cat")
        metadata = self.extractor.compute(text)

        self.assertGreater(len(metadata), len_metadata)
        self.assertEqual(metadata["word_regex_matches_count_cat"], 1)
        self.assertEqual(metadata["regex_match_count_cat"], 1)
