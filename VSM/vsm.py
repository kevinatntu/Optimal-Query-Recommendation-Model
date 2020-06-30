import argparse 
import os
import csv
import numpy as np
import sys
sys.stdout.flush()
import time
import xml.etree.ElementTree as ET
import tqdm
import pandas as pd
import argparse
import re

b = 0.75
k1 = 2.0
k3 = 1000
alpha = 1
beta = 0.03
gamma = 0.03

class VSM:
    def __init__(self, model_dir = 'model'):
        self.model_dir = model_dir
        self.vocab_path = os.path.join(self.model_dir, 'vocab.all')
        self.filelist_path = os.path.join(self.model_dir, 'file-list')
        self.invertedfile_path = os.path.join(self.model_dir , 'inverted-file')
        
        
        print('Reading files')
        with open(self.vocab_path, 'r', encoding='utf-8') as file:
            self.vocab = np.array([_v.strip() for _v in file.readlines()])
            encoding= self.vocab[0]
            if encoding.lower() != 'utf-8' and encoding.lower() != 'utf8':
                print(self.filelist_path + ' is not in utf-8 format')


        with open(self.filelist_path, 'r', encoding='utf-8') as file:
            self.filelist = np.array([_v.strip() for _v in file.readlines()])

        with open(self.invertedfile_path, 'r', encoding='utf-8') as file:    
            self.invertedfile = np.array([_v.strip() for _v in file.readlines()])


        print('Processing vocab and document file...', end='')
        self.TF = {}
        self.terms = []
        self.doc_length = np.zeros(self.filelist.shape)

        start_time = time.time()
        _line = -1
        while _line < self.invertedfile.shape[0] -1:
            _line +=1            
            vocab_id_1, vocab_id_2, N = map(int, self.invertedfile[_line].split(' '))

            # default unigram    
            term = self.vocab[vocab_id_1]
            term_length = 1

            # bigram
            if vocab_id_2 != -1:
                term += self.vocab[vocab_id_2]
                term_length += 1 


            # collecting terms appear    
            self.terms.append(term)

            # initialize tf of term
            self.TF[term] = {}

            # collecting term frequency and document length    
            for n in range(N):
                _line += 1
                file_id, term_count =  map(int, self.invertedfile[_line].split(' '))
                self.TF[term][file_id] = term_count
                self.doc_length[file_id] += term_count * term_length

        term = np.array(term)
        print(round(time.time() - start_time, 2), 'sec')


            # ## Calculate TF and IDF
        # Okapi/BM25 Doc Length Normalization
        print('Calculate TF and IDF...', end='')
        start_time = time.time()
        average_doc_length = np.mean(self.doc_length)
        N = len(self.filelist)
        for term in self.TF:
            for doc in self.TF[term]:
                self.TF[term][doc] = ((k1 + 1) * self.TF[term][doc]) / ((k1 * (1 - b + b * (self.doc_length[doc] / average_doc_length)))+self.TF[term][doc])


        self.IDF = dict.fromkeys(self.TF.keys())
        for term in self.IDF:
            # TF[term] = number of documents = df
            df_t = len(self.TF[term])
            self.IDF[term] = np.log((N - df_t+ 0.5) / (df_t + 0.5))
        print(round(time.time() - start_time, 2), 'sec')
        
    def retrieval(self, query):
        scores_follow_filelist = []
        descending_ranking_by_id = []
        query = np.array(query)
        retrieved_docs = []
        for q in query:#tqdm.tqdm(query):
            q = re.sub(r'[^\w\s]',' ',q).lower().split(' ')

            q = self.get_substring(q)
            q = self.get_valid_encoding(q)

        #     Calculate term frequancy in query, qtf
            self.QTF = {}
            (terms_, frequencies) = np.unique(q, return_counts=True)


            for index, term in enumerate(terms_):
                self.QTF[term] = frequencies[index]

            self.scores = self.get_document_score()
            
            scores_index = np.argsort(self.scores, axis=0)
            scores_index = np.argsort(self.scores, axis=0)[::-1]
            scores_follow_filelist.append(self.scores)
            descending_ranking_by_id.append(scores_index)
            
        return scores_follow_filelist, descending_ranking_by_id

            
    def check_latin(self, term):
        new_term = ''
        for term_uchar in term:
    #         range of unicode (Latin)
            if term_uchar >= u'\uff00' and term_uchar <=u'\uff5e':
                term_uchar = chr(ord(term_uchar) - ord(u'\ufee0'))

            new_term += term_uchar

        return new_term

    def get_valid_encoding(self, query_terms):
        query_terms_valid = []
        for term in query_terms:
            if term !='':
                query_terms_valid.append(self.check_latin(term))
        return query_terms_valid

    def get_substring(self, query_terms):
        terms_combs = []
        for term in query_terms:
            terms_combs.append(term)
        return terms_combs

    def get_document_score(self):
        scores = np.zeros(self.filelist.shape)
        for term in self.QTF:
            for file_id in self.TF[term]:
                scores[file_id] += self.TF[term][file_id] * self.IDF[term] * (((k3 + 1) * self.QTF[term]) / (k3 + self.QTF[term]))
        return scores