import os
import re
import string
import collections

import nltk
import emoji
import numpy as np
import pandas as pd

from tqdm import tqdm
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split

from src.utils.file_utils import load_json, save_json, save_dict

import warnings
warnings.filterwarnings("ignore")


PUNCTUATION = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + \
    '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_map = {
    "‘": "'", "₹": "e", "´": "'", "°": "", "™": "tm", "√": " sqrt ", "×": "x",
    "²": "2", "—": "-", "–": "-", "’": "'", "`": "'", '“': '"', '”': '"', '“': '"',
    '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', '−': '-',
    'β': 'beta', '∅': '', '³': '3', 'π': 'pi', '\u200b': ' ', '…': ' ... ',
    '\ufeff': '', 'करना': '', 'है': ''
}

pattern_map = {
    (r"https?:\/\/\S+|ftp:\/\/\S+|www\.\S+"
     r"([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])"): "",
    r"([\d]+:[\d]+)": "time",
    r"([\d]+\/[\d]+(/[\d]+)?)": "date",
    r"[0-9]": "",
    r"\s+": " ",
    "&gt": " ", "&lt": " ", "&amp": " "
}

def replace_emoji(text):
    return emoji.demojize(text, delimiters=("", ""))


def remove_emoji(text):
    emoji_pattern = re.compile(
        "["u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF"
           u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF"
           u"\U00002702-\U000027B0" u"\U000024C2-\U0001F251""]+",
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_non_acsii(text):
    return ''.join([x for x in text if x in string.printable])


def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'',text)


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    filtered_sentence = []   
    for word in text.split(): 
        if word not in stop_words: 
            filtered_sentence.append(word) 
    return " ".join(filtered_sentence)


def work_tokenize(text):
    return nltk.word_tokenize(text)


def stem_words(stemmer, text):
    return " ".join([stemmer.stem(word) for word in work_tokenize(text)])

def lemma_words(lemmatizer, text):
    wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
    pos_tagged_text = nltk.pos_tag(work_tokenize(text))
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN))
                     for word, pos in pos_tagged_text])


def spell_check(spell_checker, text):
    corrected_words = []
    all_words = work_tokenize(text)
    misspelled_words = spell_checker.unknown(all_words)
    for word in all_words:
        if word in misspelled_words:
            corrected_words.append(spell_checker.correction(word))
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)


def correct_data(df):
    ids_with_target_error = [328, 443, 513, 2619, 3640, 3900, 4342, 5781, 6552,
                             6554, 6570, 6701, 6702, 6729, 6861, 7226]
    df.loc[df["id"].isin(ids_with_target_error), "target"] = 0
    return df


def normalize_sentence(text, config):
    if config.get("to_lower", False):
        text = text.lower()
    for p in pattern_map:
        text = re.sub(p, pattern_map[p], text)
    text = remove_non_acsii(remove_emoji(replace_emoji(remove_html(text))))

    for p in punct_map:
        text = text.replace(p, punct_map[p])
    for p in PUNCTUATION:
        if config.get("filter_special_chars", False):
            text = text.replace(p, " ")
        else:
            text = text.replace(p, f" {p} ")

    text = re.sub("\s+", " ", text).strip()

    if config.get("remap_contraction", False):
        contraction_map = load_json("resources/contraction_map.json")
        words = text.split()
        text = " ".join(contraction_map.get(w, w) for w in words)

    if config.get("remap_abbreviation", False):
        abbreviation_map = load_json("resources/abbreviation_map.json")
        words = text.split()
        text = " ".join(abbreviation_map.get(w, w) for w in words)

    if config.get("spell_check", False):
        from spellchecker import SpellChecker

        spell_checker = SpellChecker(distance=1)
        text = spell_check(spell_checker, text)

    if config.get("stem", False):
        stemmer = PorterStemmer()
        text = stem_words(stemmer, text)

    if config.get("lemma", False):
        lemmatizer = WordNetLemmatizer()
        text = lemma_words(lemmatizer, text)

    return text


def preprocess(df, config, is_train=False):
    import dask.dataframe as ddf

    df_dask = ddf.from_pandas(df, npartitions=30)
    df["keyword"] = df_dask.map_partitions(lambda d: d["keyword"].apply(
        (lambda t: normalize_sentence(t, config)))).compute(scheduler='processes')  
    df["location"] = df_dask.map_partitions(lambda d: d["location"].apply(
        (lambda t: normalize_sentence(t, config)))).compute(scheduler='processes')
    df["text"] = df_dask.map_partitions(lambda d: d["text"].apply(
        (lambda t: normalize_sentence(t, config)))).compute(scheduler='processes')
    if is_train:
        df = correct_data(df)
    if config.get("add_metadata_to_text", False):
        df["text"] = df[["text", "location", "keyword"]].apply(
            lambda x: re.sub("\s+", " ", f"{x[0]} {x[1]} {x[2]}"), axis=1)
    return df


def build_vocab(texts, threshold=3):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = collections.defaultdict(lambda: 0)
    for sentence in sentences:
        for word in sentence:
            vocab[word] += 1

    words = list(vocab.keys())
    for k in words:
        if vocab[k] < threshold:
            vocab.pop(k)
    return dict(sorted(vocab.items(), key=lambda x: x[1], reverse=True))


def main(config_path, force=False):
    config = load_json(config_path)
    preprocess_config = config["data"]["preprocess"]

    train_data = pd.read_csv("data/raw/train.csv").fillna("")
    test_data  = pd.read_csv("data/raw/test.csv").fillna("")

    train_data = preprocess(train_data, preprocess_config, is_train=True)
    test_data  = preprocess(test_data, preprocess_config)

    data_dir = os.path.split(config["data"]["path"]["train"])[0]
    if os.path.exists(data_dir) and not force:
        raise ValueError(data_dir + " already existed")
    os.makedirs(data_dir, exist_ok=True)

    idxs = train_data.id.tolist()
    labels = train_data.target.tolist()
    train_ids, val_ids, _, _ = train_test_split(
        idxs, labels, train_size=0.8, random_state=config["seed"], stratify=labels)
    val_data = train_data[train_data.id.isin(val_ids)]
    train_data = train_data[train_data.id.isin(train_ids)]

    train_data.to_csv(config["data"]["path"]["train"], sep="\t", index=False)
    val_data.to_csv(config["data"]["path"]["val"], sep="\t", index=False)
    test_data.to_csv(config["data"]["path"]["test"], sep="\t", index=False)

    all_sentences = pd.concat([train_data["text"], test_data["text"]])
    vocab = build_vocab(all_sentences)
    save_dict(config["data"]["path"]["vocab"], vocab, sep="\t")

    save_json(os.path.join(data_dir, "config.json"), config["data"])
