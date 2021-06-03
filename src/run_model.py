#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 16:57:02 2021

@author: oceanesalmeron
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import LinearSVC
from clean_text import clean_tweet

def load_data(folder_path):
    """ 
    Load csv data to dataframe
    """
    print("> Loading Data")
    train_data = pd.read_csv(folder_path+"/Corona_NLP_train.csv")
    test_data = pd.read_csv(folder_path+"/Corona_NLP_test.csv")

    return train_data, test_data

def encode_sentiment(x):
    """
    Encoding sentiment 
    """
    if x == "Extremely Positive" or x == "Positive":
        return 1
    elif x == "Extremely Negative" or x == "Negative":
        return -1
    else:
        return 0

def create_model():
    """
    Model creation
    """
    print("> Creating Model")
    tfidf = TfidfVectorizer(tokenizer = clean_tweet)
    svm = LinearSVC()
    steps = [('tfidf',tfidf),('svm',svm)]
    pipe = Pipeline(steps)

    return pipe


if __name__ == "__main__":
    # Load data
    train, test = load_data("data")

    # Encode sentiment 
    train['Sentiment'] = train['Sentiment'].apply(lambda x: encode_sentiment(x))
    test['Sentiment'] = test['Sentiment'].apply(lambda x: encode_sentiment(x))

    # Separate train, val and test sets
    X = train['OriginalTweet']
    y = train['Sentiment']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)
    X_test = test['OriginalTweet']
    y_test = test['Sentiment']
    
    # Training model
    model = create_model()
    model.fit(X_train,y_train)

    # Predict
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    print("> Metrics for train/val sets:")
    print(classification_report(y_val,y_val_pred))
    print(confusion_matrix(y_val,y_val_pred))

    print("> Metrics for test set:")
    print(classification_report(y_test,y_test_pred))
    print(confusion_matrix(y_test,y_test_pred))

