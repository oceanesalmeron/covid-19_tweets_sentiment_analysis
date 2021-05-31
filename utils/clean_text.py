import re
import nltk
from nltk.corpus import stopwords

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

def remove_space(x):
    return re.sub(r'\s+', '', x)

def remove_stopwords(x):
    stop_words = stopwords.words('english')
    return [item for item in x.split() if item not in stop_words]

def lower(x):
    return x.lower()

def clean_tweet(x): 
    x = remove_urls(x)
    x = remove_html(x)
    x = remove_mention(x)
    x = remove_hashtags(x)
    x = remove_punctuations(x)
    x = remove_numbers(x)
    x = lower(x)
    x = remove_stopwords(x)

    return x
