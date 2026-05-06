from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

import importlib.util
import unittest

PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None

if PANDAS_AVAILABLE:
    import pandas as pd

    from src.preprocessing import (
        add_text_meta_features,
        apply_label_corrections,
        clean_categorical_value,
        preprocess_dataframe,
    )


@unittest.skipUnless(PANDAS_AVAILABLE, "pandas is required for preprocessing tests")
class PreprocessingTests(unittest.TestCase):
    def test_clean_categorical_value_fills_missing_and_empty_cleaned_values(self):
        self.assertEqual(clean_categorical_value(None, "keyword"), "no_keyword")
        self.assertEqual(clean_categorical_value("!!!", "location"), "no_location")

    def test_clean_categorical_value_removes_unstable_characters(self):
        self.assertEqual(clean_categorical_value("New-York/@NYC", "location"), "NewYorkNYC")
        self.assertEqual(clean_categorical_value("storm_warning", "keyword"), "storm_warning")

    def test_add_text_meta_features_returns_copy_with_expected_counts(self):
        source = pd.DataFrame(
            {
                "id": [1],
                "keyword": ["storm"],
                "location": ["NY"],
                "text": ["Hello @you! #storm storm"],
            }
        )

        processed = add_text_meta_features(source)

        self.assertNotIn("word_count", source.columns)
        self.assertEqual(processed.loc[0, "word_count"], 4)
        self.assertEqual(processed.loc[0, "unique_word_count"], 4)
        self.assertEqual(processed.loc[0, "char_count"], len("Hello @you! #storm storm"))
        self.assertEqual(processed.loc[0, "punctuation_count"], 3)
        self.assertEqual(processed.loc[0, "hashtag_count"], 1)
        self.assertEqual(processed.loc[0, "mention_count"], 1)

    def test_apply_label_corrections_changes_known_duplicate_label_only_on_copy(self):
        known_text = "To fight bioterrorism sir."
        source = pd.DataFrame({"text": [known_text, "unchanged"], "target": [1, 1]})

        processed = apply_label_corrections(source)

        self.assertEqual(source.loc[0, "target"], 1)
        self.assertEqual(processed.loc[0, "target"], 0)
        self.assertEqual(processed.loc[1, "target"], 1)

    def test_preprocess_dataframe_combines_categorical_features_and_label_corrections(self):
        known_text = "To fight bioterrorism sir."
        source = pd.DataFrame(
            {
                "id": [1],
                "keyword": [None],
                "location": ["São Paulo!!!"],
                "text": [known_text],
                "target": [1],
            }
        )

        processed = preprocess_dataframe(source, correct_labels=True)

        self.assertEqual(processed.loc[0, "keyword"], "no_keyword")
        self.assertEqual(processed.loc[0, "location"], "So Paulo")
        self.assertEqual(processed.loc[0, "target"], 0)
        self.assertEqual(processed.loc[0, "word_count"], 4)


if __name__ == "__main__":
    unittest.main()
