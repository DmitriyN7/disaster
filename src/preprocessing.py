"""Pure preprocessing helpers for the disaster tweets dataset."""

from collections.abc import Iterable, Mapping
from typing import Any
import string

import pandas as pd
from wordcloud import STOPWORDS

CATEGORICAL_COLUMNS = ("keyword", "location")

LABEL_CORRECTIONS: Mapping[str, int] = {
    "like for the music video I want some real action shit like burning buildings and police chases not some weak ben winston shit": 0,
    "Hellfire is surrounded by desires so be careful and don\x89Ûªt let your desires control you! #Afterlife": 0,
    "To fight bioterrorism sir.": 0,
    ".POTUS #StrategicPatience is a strategy for #Genocide; refugees; IDP Internally displaced people; horror; etc. https://t.co/rqWuoy1fm4": 1,
    "CLEARED:incident with injury:I-495  inner loop Exit 31 - MD 97/Georgia Ave Silver Spring": 1,
    "#foodscare #offers2go #NestleIndia slips into loss after #Magginoodle #ban unsafe and hazardous for #humanconsumption": 0,
    "In #islam saving a person is equal in reward to saving all humans! Islam is the opposite of terrorism!": 0,
    "Who is bringing the tornadoes and floods. Who is bringing the climate change. God is after America He is plaguing her\n \n#FARRAKHAN #QUOTE": 1,
    "RT NotExplained: The only known image of infamous hijacker D.B. Cooper. http://t.co/JlzK2HdeTG": 1,
    "Mmmmmm I'm burning.... I'm burning buildings I'm building.... Oooooohhhh oooh ooh...": 0,
    "wowo--=== 12000 Nigerian refugees repatriated from Cameroon": 0,
    "He came to a land which was engulfed in tribal war and turned it into a land of peace i.e. Madinah. #ProphetMuhammad #islam": 0,
    "Hellfire! We don\x89Ûªt even want to think about it or mention it so let\x89Ûªs not do anything that leads to it #islam!": 0,
    "The Prophet (peace be upon him) said 'Save yourself from Hellfire even if it is by giving half a date in charity.'": 0,
    "Caution: breathing may be hazardous to your health.": 1,
    "I Pledge Allegiance To The P.O.P.E. And The Burning Buildings of Epic City. ??????": 0,
    "#Allah describes piling up #wealth thinking it would last #forever as the description of the people of #Hellfire in Surah Humaza. #Reflect": 0,
    "that horrible sinking feeling when you\x89Ûªve been at home on your phone for a while and you realise its been on 3G this whole time": 0,
}


def clean_categorical_value(value: Any, column_name: str) -> str:
    """Fill and normalize a single categorical value without mutating callers."""

    placeholder = f"no_{column_name}"

    if pd.isna(value):
        return placeholder

    cleaned = pd.Series([value]).astype(str).str.replace(r"[^a-zA-Z0-9_ ]", "", regex=True).iloc[0]
    return cleaned or placeholder


def preprocess_categorical_columns(
    df: pd.DataFrame, columns: Iterable[str] = CATEGORICAL_COLUMNS
) -> pd.DataFrame:
    """Return a copy with stable categorical values for keyword/location columns."""
    processed = df.copy()
    for column in columns:
        processed[column] = processed[column].map(
            lambda value: clean_categorical_value(value, column)
        )
    return processed


def count_words(text: object) -> int:
    """Count whitespace-delimited words in text."""
    return len(str(text).split())


def count_unique_words(text: object) -> int:
    """Count unique whitespace-delimited words in text."""
    return len(set(str(text).split()))


def count_stop_words(text: object) -> int:
    """Count stop words using the wordcloud stop-word list."""
    return len([word for word in str(text).lower().split() if word in STOPWORDS])


def count_punctuation(text: object) -> int:
    """Count ASCII punctuation characters in text."""
    return len([character for character in str(text) if character in string.punctuation])


def count_character(text: object, character: str) -> int:
    """Count occurrences of one character in text."""
    return len([candidate for candidate in str(text) if candidate == character])


def add_text_meta_features(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Return a copy with deterministic text meta-features added."""
    processed = df.copy()
    text = processed[text_column]
    processed["word_count"] = text.map(count_words)
    processed["unique_word_count"] = text.map(count_unique_words)
    processed["stop_word_count"] = text.map(count_stop_words)
    processed["char_count"] = text.map(lambda value: len(str(value)))
    processed["punctuation_count"] = text.map(count_punctuation)
    processed["hashtag_count"] = text.map(lambda value: count_character(value, "#"))
    processed["mention_count"] = text.map(lambda value: count_character(value, "@"))
    return processed


def apply_label_corrections(
    df: pd.DataFrame, corrections: Mapping[str, int] = LABEL_CORRECTIONS
) -> pd.DataFrame:
    """Return a copy with known contradictory duplicate labels corrected."""
    processed = df.copy()
    if "target" not in processed.columns:
        return processed

    for text, target in corrections.items():
        processed.loc[processed["text"] == text, "target"] = target
    return processed


def preprocess_dataframe(df: pd.DataFrame, *, correct_labels: bool = False) -> pd.DataFrame:
    """Run all deterministic preprocessing steps and return a new dataframe."""
    processed = preprocess_categorical_columns(df)
    processed = add_text_meta_features(processed)
    if correct_labels:
        processed = apply_label_corrections(processed)
    return processed
