#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 06:58:00 2021

@author: jaisonjohn
"""

from rouge import Rouge
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from urllib.parse import urlencode
import urllib.request
import gzip
import json

rouge = Rouge()

#ROUGE Score
def calculateRougeScores(model_summary, gold_summary):
    '''Calculate the Rouge Score'''
    
    return rouge.get_scores(model_summary, gold_summary)


#Cosine Similarity
def cosine_similarity_metric(model_summary, gold_summary, embed_model_name):
    '''Calculate the Cosine Similarity between two Summaries'''
    
    model = SentenceTransformer(embed_model_name)
    ref_embedding = model.encode(gold_summary)
    model_embedding = model.encode(model_summary)
    return (1-cosine(ref_embedding, model_embedding))

#NER similarity
def NER_similarity_score(model_summary, gold_summary):
    '''Calculate the Named Entities overlapping score'''
    
    print('NER Model Summary', model_summary)
    model_synsetids = get_synsetids(model_summary)
    print('NER Gold Summary', gold_summary)
    gold_synsetids = get_synsetids(gold_summary)
    score = len(set(model_synsetids) & set(gold_synsetids)) / len(set(model_synsetids + gold_synsetids))
    
    return score

def get_synsetids(text):
    key = '1a79449a-d882-4daa-9f94-9a3f6090aa5f'
    lang = 'EN'
    annType='NAMED_ENTITIES'
    params = {
        'text' : text,
        'lang' : lang,
        'annType':annType,
        'key'  : key
        }
    service_url = 'https://babelfy.io/v1/disambiguate'
    url = service_url + '?' + urlencode(params)
    data = GET_data(url)
    synsetId_list = []
    for result in data:
        
        # retrieving char fragment
        charFragment = result.get('charFragment')
        cfStart = charFragment.get('start')
        cfEnd = charFragment.get('end')
        print(cfStart, cfEnd, text[cfStart:cfEnd+1], sep='\t')
    
        synsetId_list.append(result.get('babelSynsetID'))
    return synsetId_list
    
def GET_data(url):
    HTTP_HEADERS={
        #'User-Agent': 'Mozilla/5.0 Firefox/34.0',
        #'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-encoding':'gzip'
        }
    req=urllib.request.Request(url,None,HTTP_HEADERS)
    response=urllib.request.urlopen(req)

    if response.info().get('Content-Encoding') == 'gzip':
        result_bytes = gzip.decompress(response.read())
        #result_json_string = result_bytes.decode('utf8').replace("'", '"').replace('false','False').replace('true','True')
        #data = eval(result_json_string)
        data=json.loads(result_bytes.decode("UTF-8"))
        return data
    else:
        raise Exception('URL did not return gzip')

#Plot Heatmap
def plot_heatmap(data_matrix, title = None, 
                 xticklabels = None,yticklabels = None,
                 xlabel = None,
                 ylabel = None):
    '''Plot a heatmap using the input data'''
    
    sns.heatmap(data_matrix,xticklabels = xticklabels,yticklabels = yticklabels,vmin=0, vmax=1, annot = True)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel) 
    if ylabel is not None:
        plt.ylabel(ylabel) 
    plt.show()

# def evaluate(evaluation_methods,evaluation_summaries,rouge1 = True, rouge2 = True):
    
#     embed_model_name = 'stsb-roberta-base'
#     rouge1_metric = np.zeros((len(evaluation_summaries),len(evaluation_summaries)))
#     rouge2_metric = np.zeros((len(evaluation_summaries),len(evaluation_summaries)))
#     embed_metric = np.zeros((len(evaluation_summaries),len(evaluation_summaries)))
#     for i, summary1 in enumerate(evaluation_summaries):
#         for j, summary2 in enumerate(evaluation_summaries):
#             rougeScore = calculateRougeScores(summary1,summary2)
#             rouge1_metric[i,j] = rougeScore[0]['rouge-1']['f']
#             rouge2_metric[i,j] = rougeScore[0]['rouge-2']['f']
            
#             embed_score = cosine_similarity_metric(summary1, summary2, embed_model_name)
#             embed_metric[i,j] = embed_score
    
    
    
                
#     return rouge1_metric, rouge2_metric, embed_metric


def evaluate2(evaluation_methods,
              evaluation_summaries,
              evaluation_metrices,
              rouge1 = True, 
              rouge2 = True):
    
    #evaluation_metrices = ['Rouge-1','Rouge-2','Embed Metric-1','Embed Metric-2','Embed Metric-3']
    embed_model_name1 = 'stsb-roberta-base'
    embed_model_name2 = 'LaBSE'
    embed_model_name3 = 'xlm-r-100langs-bert-base-nli-stsb-mean-tokens'
    
    evaluation_scores = np.zeros((len(evaluation_summaries),len(evaluation_metrices),len(evaluation_summaries)-1))
    evaluation_methods_out = []
    
    for i, summary1 in enumerate(evaluation_summaries):
        methods_out = []
        n = 0
        for j, summary2 in enumerate(evaluation_summaries):
            if i != j:
                rougeScore = calculateRougeScores(summary1,summary2)
                evaluation_scores[i,0,n] = rougeScore[0]['rouge-1']['f']
                evaluation_scores[i,1,n] = rougeScore[0]['rouge-2']['f']
                evaluation_scores[i,2,n] = cosine_similarity_metric(summary1, summary2, embed_model_name1)
                evaluation_scores[i,3,n] = cosine_similarity_metric(summary1, summary2, embed_model_name2)
                evaluation_scores[i,4,n] = cosine_similarity_metric(summary1, summary2, embed_model_name3)
                evaluation_scores[i,5,n] = NER_similarity_score(summary1,summary2)
                
                methods_out.append(evaluation_methods[j])
                n+=1
        evaluation_methods_out.append(methods_out)
    
    return evaluation_scores,evaluation_methods_out
                
                
            
            