#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 08:25:15 2021

@author: jaisonjohn
"""

import utilities
import math

class tfidfSummarizer():
    
    # def __init__(self):
    #     pass
    
    def create_tf_matrix(self, sentences):
        """
        Here document refers to a sentence.
        TF(t) = (Number of times the term t appears in a document) / 
        (Total number of terms in the document)
        """
        print('Creating tf matrix.')
        tf_matrix = {}
        for sentence in sentences:
            tf_table = {}
    
            words_count = len(sentence)
            clean_words = utilities.text_preprocessing([sentence])
    
            # Determining frequency of words in the sentence
            word_freq = {}
            for word in clean_words:
                word_freq[word] = (word_freq[word] + 1) if word in word_freq else 1
    
            # Calculating tf of the words in the sentence
            for word, count in word_freq.items():
                tf_table[word] = count / words_count
    
            tf_matrix[sentence[:15]] = tf_table

        return tf_matrix
    
    def create_idf_matrix(self, sentences):
        """
        IDF(t) = log_e(Total number of documents / Number of documents with term t in it)
        """
        print('Creating idf matrix.')
    
        idf_matrix = {}
    
        documents_count = len(sentences)
        sentence_word_table = {}
    
        # Getting words in the sentence
        for sentence in sentences:
            clean_words = utilities.text_preprocessing([sentence])
            sentence_word_table[sentence[:15]] = clean_words
    
        # Determining word count table with the count of sentences which contains the word.
        word_in_docs = {}
        for sent, words in sentence_word_table.items():
            for word in words:
                word_in_docs[word] = (word_in_docs[word] + 1) if word in word_in_docs else 1
    
        # Determining idf of the words in the sentence.
        for sent, words in sentence_word_table.items():
            idf_table = {}
            for word in words:
                idf_table[word] = math.log10(documents_count / float(word_in_docs[word]))
    
            idf_matrix[sent] = idf_table
    
        return idf_matrix
    
    def create_tf_idf_matrix(self, tf_matrix, idf_matrix):
        """
        Create a tf-idf matrix which is multiplication of tf * idf individual words
        """
        print('Calculating tf-idf of sentences.')
    
        tf_idf_matrix = {}
    
        for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
            tf_idf_table = {}
    
            for (word1, value1), (word2, value2) in zip(f_table1.items(), f_table2.items()):
                tf_idf_table[word1] = float(value1 * value2)
    
            tf_idf_matrix[sent1] = tf_idf_table
    
        return tf_idf_matrix
    
    def create_sentence_score_table(self, tf_idf_matrix):
        """
        Determining average score of words of the sentence with its words tf-idf value.
        """
        print('Creating sentence score table.')
    
        sentence_value = {}
    
        for sent, f_table in tf_idf_matrix.items():
            total_score_per_sentence = 0
    
            count_words_in_sentence = len(f_table)
            for word, score in f_table.items():
                total_score_per_sentence += score
    
            if count_words_in_sentence == 0:
                sentence_value[sent] = 0
            else:
                sentence_value[sent] = total_score_per_sentence / count_words_in_sentence
    
        return sentence_value
    
    def find_average_score(self, sentence_value):
        """
        Calculate average value of a sentence form the sentence score table.
        """
        print('Finding average score')
    
        sum = 0
        for val in sentence_value:
            sum += sentence_value[val]
    
        average = sum / len(sentence_value)
    
        return average
    
    
    def generate_summary(self, sentences, sentence_value, threshold):
        """
        Generate a sentence for sentence score greater than average.
        """
        print('Generating summary')
        #print(sentence_value)
        sentence_count = 0
        summary = ''
        sum_tmp = []
        
        #sentences = np.unique(sentences)
        sl = int(len(sentences)*0.05)
        #sl=1
        #keymax = max(sentence_value, key=sentence_value.get)
        keymax = sorted(sentence_value, key=sentence_value.get, reverse=True)[:sl]
        print(keymax)
        print(type(sentences))
        
        for sentence in sentences:
            #if sentence[:15] in sentence_value and sentence_value[sentence[:15]] >= threshold:
            if sentence[:15] in keymax and sentence[:15] not in sum_tmp:
                sum_tmp.append(sentence)
                #summary += sentence + " "
                sentence_count += 1
                if sentence_count == len(keymax):
                    break
                
        summary = ' '.join([sent for sent in sum_tmp])
        return summary