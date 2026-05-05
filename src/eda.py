import string

from wordcloud import STOPWORDS
import numpy as np
import pandas as pd

df_train = pd.read_csv("data/train.csv", dtype={"id": np.int16, "target": np.int8})
df_test = pd.read_csv("data/test.csv", dtype={"id": np.int8})

# PREPROCESSING
cols_to_process = ["keyword", "location"]

# fill NAs
for df in (df_train, df_test):
    for col in cols_to_process:
        df[col] = df[col].fillna(f"no_{col}")

# clean text
for df in (df_train, df_test):
    for col in cols_to_process:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[^a-zA-Z0-9_ ]", "", regex=True)
            .replace("", f"no_{col}")
        )

# Meta features
df_train["word_count"] = df_train["text"].apply(lambda x: len(str(x).split()))
df_test["word_count"] = df_test["text"].apply(lambda x: len(str(x).split()))

df_train["unique_word_count"] = df_train["text"].apply(lambda x: len(set(str(x).split())))
df_test["unique_word_count"] = df_test["text"].apply(lambda x: len(set(str(x).split())))

# word like "the", "a" etc.
df_train["stop_word_count"] = df_train["text"].apply(
    lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS])
)
df_test["stop_word_count"] = df_test["text"].apply(
    lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS])
)

df_train["char_count"] = df_train["text"].apply(lambda x: len(str(x)))
df_test["char_count"] = df_test["text"].apply(lambda x: len(str(x)))

df_train["punctuation_count"] = df_train["text"].apply(
    lambda x: len([c for c in str(x) if c in string.punctuation])
)
df_test["punctuation_count"] = df_test["text"].apply(
    lambda x: len([c for c in str(x) if c in string.punctuation])
)

df_train["hashtag_count"] = df_train["text"].apply(lambda x: len([c for c in str(x) if c == "#"]))
df_test["hashtag_count"] = df_test["text"].apply(lambda x: len([c for c in str(x) if c == "#"]))

df_train["mention_count"] = df_train["text"].apply(lambda x: len([c for c in str(x) if c == "@"]))
df_test["mention_count"] = df_test["text"].apply(lambda x: len([c for c in str(x) if c == "@"]))

# remapping duplicates
df_train.loc[
    df_train["text"]
    == "like for the music video I want some real action shit like burning buildings and police chases not some weak ben winston shit",
    "target",
] = 0
df_train.loc[
    df_train["text"]
    == "Hellfire is surrounded by desires so be careful and donÛªt let your desires control you! #Afterlife",
    "target",
] = 0
df_train.loc[df_train["text"] == "To fight bioterrorism sir.", "target"] = 0
df_train.loc[
    df_train["text"]
    == ".POTUS #StrategicPatience is a strategy for #Genocide; refugees; IDP Internally displaced people; horror; etc. https://t.co/rqWuoy1fm4",
    "target",
] = 1
df_train.loc[
    df_train["text"]
    == "CLEARED:incident with injury:I-495  inner loop Exit 31 - MD 97/Georgia Ave Silver Spring",
    "target",
] = 1
df_train.loc[
    df_train["text"]
    == "#foodscare #offers2go #NestleIndia slips into loss after #Magginoodle #ban unsafe and hazardous for #humanconsumption",
    "target",
] = 0
df_train.loc[
    df_train["text"]
    == "In #islam saving a person is equal in reward to saving all humans! Islam is the opposite of terrorism!",
    "target",
] = 0
df_train.loc[
    df_train["text"]
    == "Who is bringing the tornadoes and floods. Who is bringing the climate change. God is after America He is plaguing her\n \n#FARRAKHAN #QUOTE",
    "target",
] = 1
df_train.loc[
    df_train["text"]
    == "RT NotExplained: The only known image of infamous hijacker D.B. Cooper. http://t.co/JlzK2HdeTG",
    "target",
] = 1
df_train.loc[
    df_train["text"]
    == "Mmmmmm I'm burning.... I'm burning buildings I'm building.... Oooooohhhh oooh ooh...",
    "target",
] = 0
df_train.loc[
    df_train["text"] == "wowo--=== 12000 Nigerian refugees repatriated from Cameroon",
    "target",
] = 0
df_train.loc[
    df_train["text"]
    == "He came to a land which was engulfed in tribal war and turned it into a land of peace i.e. Madinah. #ProphetMuhammad #islam",
    "target",
] = 0
df_train.loc[
    df_train["text"]
    == "Hellfire! We donÛªt even want to think about it or mention it so letÛªs not do anything that leads to it #islam!",
    "target",
] = 0
df_train.loc[
    df_train["text"]
    == "The Prophet (peace be upon him) said 'Save yourself from Hellfire even if it is by giving half a date in charity.'",
    "target",
] = 0
df_train.loc[
    df_train["text"] == "Caution: breathing may be hazardous to your health.", "target"
] = 1
df_train.loc[
    df_train["text"]
    == "I Pledge Allegiance To The P.O.P.E. And The Burning Buildings of Epic City. ??????",
    "target",
] = 0
df_train.loc[
    df_train["text"]
    == "#Allah describes piling up #wealth thinking it would last #forever as the description of the people of #Hellfire in Surah Humaza. #Reflect",
    "target",
] = 0
df_train.loc[
    df_train["text"]
    == "that horrible sinking feeling when youÛªve been at home on your phone for a while and you realise its been on 3G this whole time",
    "target",
] = 0
