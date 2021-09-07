#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 07:45:51 2021

@author: jaisonjohn
"""

import numpy as np
import utilities
import TFIDFSummarizer
import LSASummarizer
import ClusteringSummarizer
import DynESummarizer
import Evaluation
import AutoEncoder
import torch
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
from transformers import BartTokenizer
import time
import nltk
# nltk.download('punkt')
#nltk.download('stopwords')

start_time = time.time()

no_of_clusters = 11
no_of_articles_test = 5

file = 'val.jsonl'
dataset = utilities.data_loader(file, no_of_clusters)

summary = dataset[no_of_clusters-1]['summary']

#Join the text from multiple articles in the cluster
text = utilities.join_text(dataset[no_of_clusters-1], no_of_articles = no_of_articles_test)
# text = '''
# he is good. He is bad.
# '''

#Create list of sentences from text
sentences = utilities.split_sentences(text)

evaluation_methods = []
evaluation_summaries = []
evaluation_methods.append('Gold')
evaluation_summaries.append(summary)

#-----------------------------------------------------------------------------
#TFIDF Summarization
print('TFIDF Started')
tfidf = TFIDFSummarizer.tfidfSummarizer()
tf_matrix = tfidf.create_tf_matrix(sentences)
idf_matrix = tfidf.create_idf_matrix(sentences)
tf_idf_matrix = tfidf.create_tf_idf_matrix(tf_matrix, idf_matrix)
sentence_value = tfidf.create_sentence_score_table(tf_idf_matrix)
threshold = tfidf.find_average_score(sentence_value)
tfidfSummary = tfidf.generate_summary(sentences, sentence_value, threshold)

evaluation_methods.append('TFIDF')
evaluation_summaries.append(tfidfSummary)
print('TFIDF Ended')
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#LSA Summarization
print('LSA Started')
lsa = LSASummarizer.lsaSummarizer()
lsaSummary = lsa(text,2)

evaluation_methods.append('LSA')
evaluation_summaries.append(lsaSummary)
print('LSA Ended')
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
#Embedding and Clustering
print('Embedding Cluster Started')
clust_embed = ClusteringSummarizer.Embed_cluster_Summarizer()
clust_embed_summaries = clust_embed.generate_summary(sentences)

evaluation_methods.append('Embedd_cluster:1')
evaluation_summaries.append(clust_embed_summaries[0])
evaluation_methods.append('Embedd_cluster:2')
evaluation_summaries.append(clust_embed_summaries[1])
evaluation_methods.append('Embedd_cluster:3')
evaluation_summaries.append(clust_embed_summaries[2])
print('Embedding Cluster Ended')
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# #Auto Encoder
print('Embedding/Auto Encoder Started')
X = np.array([])
Y = np.array([])
for i in range(no_of_clusters-1):
    X = np.append(X, utilities.join_text(dataset[i]))
    Y = np.append(Y, dataset[i]['summary'])
autoencoder = AutoEncoder.AutoEncoder()
autoencoder.train(X,Y)
Xtest = utilities.join_text(dataset[no_of_clusters-1], no_of_articles_test)
autoencoderSummary = autoencoder.predict(Xtest)

evaluation_methods.append('AutoEncoder')
evaluation_summaries.append(autoencoderSummary)
print('Embedding/Auto Encoder Ended')
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#Dynamic Ensembling
print('DynE Started')
model_name = "facebook/bart-large-cnn"

dyne = DynESummarizer.dyneSummarizer(model_name)
train_data, val_data = dyne.prepare_data(dataset[1],dataset[2])
trainedModel = dyne.fine_tune(train_data,val_data)

test_texts = []
for doc in dataset[no_of_clusters-1]['articles'][:no_of_articles_test]:
    test_texts.append(doc['text'])

print('Loading model')
# model = DynESummarizer.BartMultiDocumentSummariser(model_name=model_name)
model = DynESummarizer.BartMultiDocumentSummariser(model_name=model_name,trainedModel=trainedModel)
print('Model Loaded')

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
encoder_input_ids = tokenizer(test_texts, return_tensors="pt", padding="longest", truncation=True).input_ids
decoder_input_ids = torch.tensor([tokenizer.bos_token_id], dtype=torch.int64).unsqueeze(0)
print('Generating Summaries')
sequences = model.generate(
        encoder_input_ids, decoder_input_ids, beam_width=2, num_beams=2, maxlen=100, no_repeat_ngram_size=3
    )
dyneSummary = tokenizer.decode(sequences[0][0].tolist()[0],skip_special_tokens = True)

evaluation_methods.append('DynE')
evaluation_summaries.append(dyneSummary)
print('DynE Ended')
#-----------------------------------------------------------------------------


#Evaluate the summaries using different metrices
evaluation_metrices = ['Rouge-1','Rouge-2','Embed Metric-1','Embed Metric-2','Embed Metric-3','NER']
evaluation_scores,evaluation_methods_out = Evaluation.evaluate2(evaluation_methods, 
                                                                evaluation_summaries,
                                                                evaluation_metrices)
for n, evaluation_score in enumerate(evaluation_scores):
    Evaluation.plot_heatmap(evaluation_score, 
                            title = evaluation_methods[n], 
                            xticklabels =evaluation_methods_out[n],
                            yticklabels =evaluation_metrices,
                            xlabel = "Summarization Techniques", 
                            ylabel = "Evaluation Techniques")




for cnt in range(len(evaluation_summaries)):
    print('*'*50)
    print(evaluation_methods[cnt],':')
    print(evaluation_summaries[cnt])

print("--- %s minutes ---" % ((time.time() - start_time)/60))

