#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 23:25:50 2021

@author: jaisonjohn
"""

import ClusteringSummarizer
import numpy as np
import nltk
import pandas as pd
from sentence_transformers import SentenceTransformer

from keras.layers import Input, Dense
from keras.models import Model

class AutoEncoder():
    def __init__(self):
        self.encoding_dim = 384
        self.input = Input(shape=(768,))
        self.encoded = Dense(units=self.encoding_dim, activation='relu')(self.input)
        
        self.decoded = Dense(units=768, activation='sigmoid')(self.encoded)
        self.autoencoder=Model(self.input, self.decoded)
        
        self.clustSumm = ClusteringSummarizer.Embed_cluster_Summarizer()
        self.embedding = 'stsb-roberta-base'
        self.clustSumm.model = SentenceTransformer(self.embedding)
        
    def train(self,X,Y):
        
        Xtrain = np.array([])
        Ytrain = np.array([])
        for i, text in enumerate(X):
            sentences=nltk.sent_tokenize(text)
            sentences = [sentence.strip() for sentence in sentences]
            data_x = pd.DataFrame(sentences)
            data_x.columns=['sentence']
            data_x['embed_vect'] = data_x['sentence'].apply(self.clustSumm.get_sentence_embeddings)
            x = np.mean(data_x['embed_vect'])
            if np.size(Xtrain) == 0:
                Xtrain = np.reshape(x,(1,768))
            else:
                Xtrain = np.vstack((Xtrain, np.reshape(x,(1,768))))
            
            sentences=nltk.sent_tokenize(Y[i])
            sentences = [sentence.strip() for sentence in sentences]
            data_y = pd.DataFrame(sentences)
            data_y.columns=['sentence']
            data_y['embed_vect'] = data_y['sentence'].apply(self.clustSumm.get_sentence_embeddings)
            y = np.mean(data_y['embed_vect'])
            if np.size(Ytrain) == 0:
                Ytrain = np.reshape(y,(1,768))
            else:
                Ytrain = np.vstack((Ytrain, np.reshape(y,(1,768))))
        
        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        
        self.autoencoder.fit(Xtrain, Ytrain,
                epochs=50,
                batch_size=256,
                shuffle=True,
                verbose=1,
                validation_data=(Xtrain, Ytrain))
        
        
    def predict(self,Xtest):
        sentences=nltk.sent_tokenize(Xtest)
        data_x = pd.DataFrame(sentences)
        data_x.columns=['sentence']
        data_x['embeddings'] = data_x['sentence'].apply(self.clustSumm.get_sentence_embeddings)
        x = np.mean(data_x['embeddings'])
        Xtest = np.reshape(x,(1,768))
            
        # sentences=nltk.sent_tokenize(Ytest)
        # data_y = pd.DataFrame(sentences)
        # data_y.columns=['sentence']
        # data_y['embed_vect'] = data_y['sentence'].apply(ClusteringSummarizer.get_sentence_embeddings)
        # y = np.mean(data_y['embed_vect'])
        # Ytest = np.reshape(y,(1,768))
        
        predicted = self.autoencoder.predict(Xtest)
        data_x['centroid']=data_x['embeddings'].apply(lambda x: predicted[0])
        data_x['distance_from_centroid'] = data_x.apply(self.clustSumm.distance_from_centroid, axis=1)
        
        out = np.array(data_x.sort_values('distance_from_centroid',ascending = True)['sentence'])
        
        return out[0]
    