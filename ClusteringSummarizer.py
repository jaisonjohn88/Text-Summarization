#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 00:30:29 2021

@author: jaisonjohn
"""

import nltk
from scipy.spatial import distance_matrix
from nltk.cluster import KMeansClusterer
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import utilities


class Embed_cluster_Summarizer():
    def __init__(self):
        self.model = None
        
    def get_sentence_embeddings(self,sentence):
        #model = SentenceTransformer(embedding)
        embedding = self.model.encode([sentence])
        return embedding[0]
    
    def distance_from_centroid(self,row):
        #type of emb and centroid is different, hence using tolist below
        return distance_matrix([row['embeddings']], [row['centroid'].tolist()])[0][0]
    
    def generate_summary(self,sentences, NUM_CLUSTERS = 2, ITERATIONS = 25):
        #Different Embedding Techniques
        embeddings = ['stsb-roberta-base',
                 'LaBSE',
                 'xlm-r-100langs-bert-base-nli-stsb-mean-tokens']
        
        kclusterer = KMeansClusterer(
            NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance,
            repeats=ITERATIONS,avoid_empty_clusters=True)
        
        embedding_vectors = []
        summary_list = []
        data = pd.DataFrame(sentences)
        data.columns=['sentence']
        
        for embedding in embeddings:
            self.model = SentenceTransformer(embedding)
            #embedding_vectors = np.append(embedding_vectors, data['sentence'].apply(get_sentence_embeddings))
            # embedding_vectors.append(data['sentence'].apply(self.get_sentence_embeddings, args = ((embedding))))
            embedding_vectors.append(data['sentence'].apply(self.get_sentence_embeddings))
            
        for i, embedding_vector in enumerate(embedding_vectors):
            data['embeddings'] = embedding_vector
            X = np.array(data['embeddings'].tolist())
            assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
            data['cluster']=pd.Series(assigned_clusters, index=data.index)
            data['centroid']=data['cluster'].apply(lambda x: kclusterer.means()[x])
            data['distance_from_centroid'] = data.apply(self.distance_from_centroid, axis=1)
            summary=' '.join(data.sort_values('distance_from_centroid',ascending = True)
                         .groupby('cluster')
                         .head(1)
                         .sort_index()['sentence']
                         .tolist())
            summary_list.append(summary)
            print('-'*100)
            print('Summary using',embeddings[i], ':')
            print(summary)
        
        return summary_list