import re
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])  

def remove_punctuations(x):
    return re.sub(r'[^\w\s]', '', x)

def remove_numbers(x):
    return re.sub(r'\d+', '', x)

def remove_urls(x):
    return re.sub(r"http\S+", "", x)

def remove_html(x):
    return re.sub(r'<.*?>', '', x)

def remove_mention(x):
    return re.sub(r'@\w+', '', x)

def remove_hashtags(x):
    return re.sub(r'#\w+', '', x)

def lower(x):
    return x.lower()

def remove_stopwords(x):
    return [item for item in x.split() if item not in STOP_WORDS]

def tokenization(x):
    doc = nlp(x)

    tokens = []
    for token in doc:
        if token.lemma_ != "-PRON-":
            temp = token.lemma_.strip()
        else:
            temp = token
        tokens.append(temp)

    return [token for token in tokens if token not in STOP_WORDS and token not in string.punctuation]


def clean_tweet(x): 
    x = remove_urls(x)
    x = remove_html(x)
    x = remove_mention(x)
    x = remove_hashtags(x)
    x = remove_numbers(x)
    x = lower(x)
    x = tokenization(x)

    return x