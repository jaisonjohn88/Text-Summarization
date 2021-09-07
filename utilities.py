#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 07:35:42 2021

@author: jaisonjohn
"""
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math
from scipy.spatial import distance_matrix

ps = PorterStemmer()

def data_loader(file, no_of_clusters = None):
    '''
    To load the data from the file
    '''
    dataset = [json.loads(cluster) for cluster in open(file)]
    if no_of_clusters is not None:
        dataset = dataset[:no_of_clusters]
    return dataset

def join_text(cluster, no_of_articles = None):
    
    '''
    Join the text from different articles in the cluster
    '''
    
    if no_of_articles is None:
        articles = cluster['articles']
    else:
        articles = cluster['articles'][:no_of_articles]
        
    text = ' '.join([article['text'] for article in articles])
    text = text.replace('\n', ' ')
    return text

def split_sentences(text):
    ''''
    Split the text into sentences
    '''
    return sent_tokenize(text)

def text_preprocessing(sentences):
    """
    Pre processing text to remove unnecessary words.
    """

    stop_words = set(stopwords.words('english'))

    clean_words = None
    for sent in sentences:
        words = word_tokenize(sent)
        #words = [PorterStemmer.stem(word.lower()) for word in words if word.isalnum()]
        words = [ps.stem(word.lower()) for word in words if word.isalnum()]
        clean_words = [word for word in words if word not in stop_words]

    return clean_words

