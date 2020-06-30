import time
import argparse
import math
import os
import torch
import torch.nn as nn
#import matplotlib
#from matplotlib import pyplot as plt
from transformers import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
import pickle
import numpy as np
import logging
import random
from pathlib import Path
from sklearn.metrics import mean_squared_error
import itertools
import heapq as hq
from sklearn.preprocessing import normalize
import re

from data import BERTDataset
from model import train

torch.manual_seed(66) # 42
np.random.seed(66) #numpy
random.seed(66) 

# Model setting
PAD_TOKEN = 0
UNK_TOKEN = 1
NUM_LABELS = 4
PRETRAINED_MODEL_NAME = "bert-large-cased"

model = None
tokenizer = None
model_data = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
TODO:
remove stopwords (by rule)
remove duplicates
'''

def _parse_args():
    parser = argparse.ArgumentParser(
        description="IR Final - BERT method"
    )
    parser.add_argument('--mode', type=str, help='Train or Test (Default test)', default='test') 
    parser.add_argument('--docid', type=int, help="Target DOC", nargs='+', default=[])
    parser.add_argument('--N', type=int, help="Query max length", default=5, required=True)
    parser.add_argument('--topk_query', type=int, help="Output top k query and their voting query", default=1)
    parser.add_argument('--topk_rank', action="store_true", help="Top-K ranking use")
    parser.add_argument('--random_num', type=int, help="Number of random sentences", default=500)
    parser.add_argument('--random', action="store_true", help='select by random, not in sequence')
    parser.add_argument('--output_path', type=Path, help="Output latent path", default='bert_best_query.txt')
    parser.add_argument('--model_save_path', type=Path, help="Training model save path", default='../model/')
    
    args = parser.parse_args()
    return args

def model_init():
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, hidden_dropout_prob=0.3, num_labels=NUM_LABELS, output_hidden_states=True)

    if args.mode == 'test':
        model_path = '../model/bertmodel_for_test.pkl'
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()

    return tokenizer, model

def predict():
    testData = model_data[0]
    '''
    dataloader = DataLoader(testData, 
                            batch_size=1,
                            shuffle=False, 
                            collate_fn=testData.collate_fn,
                            num_workers=4)
    '''
    trange = tqdm(enumerate(testData), total=len(testData), desc='Predict', file=sys.stdout)
    
    min_mse = 1e9
    mis_mse_lst = []
    min_latent = []
    best_query = ''
    best_query_idx = []
    best_query_lst = []
    origin_latent = None
    prev_docID = -1

    nextData = []
    origin_latent_lst = []

    query_mse_record = []
    query_text_record = []

    with torch.no_grad():
        for i, data in trange:
            tokens_tensor, segments_tensor, docID, origin_text, text_idx = data
            tokens_tensor = tokens_tensor.unsqueeze(0)
            '''
            segments_tensor = segments_tensor.unsqueeze(0)
            masks_tensors = torch.zeros(tokens_tensor.shape, 
                                    dtype=torch.long)
            masks_tensors = masks_tensors.masked_fill(tokens_tensor != 0, 1)
            '''
            outputs = model.bert(input_ids=tokens_tensor.to(device), 
                            #token_type_ids=segments_tensor.to(device), 
                            #attention_mask=masks_tensors.to(device)
                            #, labels=labels
                            )

            latent = outputs[1][0].cpu().detach().numpy()

            #print("Latent shape: ", latent.shape)
            if not args.topk_rank:
                if docID != prev_docID:
                    prev_docID = docID
                    origin_latent = latent.copy()
                    if i != 0:
                        best_query_lst[-1].append(best_query)
                        best_query_lst[-1].append(best_query_idx)
                    best_query = ''
                    best_query_lst.append([docID])
                    min_mse = 1e9
                    best_query_idx = []

                    print("DocID: {}, Origin text: {}".format(docID, origin_text))

                # calculate similarity
                # currently, use MSE
                else:
                    current_latent = latent.copy()
                    mse = mean_squared_error(origin_latent, current_latent)
                    query = origin_text
                    print("Query candidate {}: {}".format(i, query))
                    print("MSE: {}".format(mse))

                    if mse < min_mse:
                        print("Best MSE, ", mse)
                        min_mse = mse
                        min_latent = current_latent.copy()
                        best_query = query
                        best_query_idx = text_idx.copy()
            else:
                if len(origin_latent_lst) < len(args.docid):
                    origin_latent_lst.append(latent.copy())
                    if not best_query_lst:
                        best_query_lst.append([-1])
                else:
                    current_latent = latent.copy()
                    #mse = 0
                    mse_lst = []
                    for origin_latent in origin_latent_lst:
                        mse_lst.append(mean_squared_error(origin_latent, current_latent))
                    query = origin_text
                    query_text_record.append(query)
                    query_mse_record.append(mse_lst.copy())
                    #print("Query candidate {}: {}".format(i, query))
                    #print("MSE: {}".format(mse))
                    '''
                    if mse < min_mse:
                        print("Best MSE, ", mse)
                        min_mse = mse
                        min_latent = current_latent.copy()
                        best_query = query
                        best_query_idx = text_idx.copy()
                    '''
    # normalize each latent's mse first, then sum up
    if args.topk_rank:
        pass
        query_mse_record = np.asarray(query_mse_record)
        #print(query_mse_record.shape)
        mse_normalize = normalize(query_mse_record, axis=0)
        mse_normalize = np.sum(mse_normalize, axis=1)
        #print(mse_normalize.shape)
        max_query_idx = np.argmax(mse_normalize)
        min_mse = mse_normalize[max_query_idx]
        best_query = query_text_record[max_query_idx]        
        #print("Best MSE {}, query: {}".format(min_mse, best_query))

    # last doc
    best_query_lst[-1].append(best_query)
    best_query_lst[-1].append(best_query_idx)
    return best_query_lst        


if __name__ == '__main__':
    args = _parse_args()
    loglevel = os.environ.get('LOGLEVEL', 'DEBUG').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    
    logging.debug("START")

    logging.debug("Load model...")
    tokenizer, model = model_init()
    logging.debug("END")

    logging.debug("Load data...")

    if args.mode == 'train':
        trainData = BERTDataset(args.mode, '../dataset/trainset.csv', tokenizer, PAD_TOKEN)
        validData = BERTDataset(args.mode, '../dataset/validset.csv', tokenizer, PAD_TOKEN)

        model_data.append(trainData)
        model_data.append(validData)

        if not os.path.exists(args.model_save_path):
            os.mkdir(args.model_save_path)
    else:
        testData_raw = BERTDataset("test", '../dataset/task2_trainset.csv', tokenizer, PAD_TOKEN)
        # select by docID
        #testData_raw = testData_raw[args.docid]
        print(len(testData_raw))

        #select_N = args.N

        testData = []
        subData = []
        
        if not args.docid:
            print("Get 100 doc from filelist")
            with open('./file-list', 'r', encoding='utf-8') as file:
                args.docid = np.array([ int(re.search('D([0-9]+)', _v.strip()).group(1)) for _v in file.readlines()]).tolist()

        # Get substring abstract
        if not args.topk_rank:
            for docid in args.docid:
                print(docid)
                tokens_tensor, segments_tensor, _, docID, origin_text = testData_raw[docid-1]
                text_split = np.asarray(origin_text.split())
                if args.random:
                    #comb = list(itertools.combinations(list(range(len(text_split))), args.N))
                    r = list(range(len(text_split)))
                    comb = [random.choices(r, k=args.N) for _ in range(args.random_num)]
                else:
                    comb = [list(range(i, i+args.N)) for i in range(len(text_split)-args.N)]
                
                # print(comb)
                # origin
                testData.append([tokens_tensor, segments_tensor, docID, origin_text, []])
                for com in comb:
                    #print(tokens_tensor[com])
                    this_text = ' '.join(text_split[sorted(com)])
                    # Remove . & ,
                    '''
                    this_text = this_text.replace(',', '')
                    this_text = this_text.replace('.', '')
                    this_text = this_text.replace('(', '')
                    this_text = this_text.replace(')', '')
                    this_text = this_text.replace('[', '')
                    this_text = this_text.replace(']', '')
                    this_text = this_text.replace(':', '')
                    this_text = this_text.replace(';', '')
                    '''
                    #this_text = this_text.replace('"', '')
                    #this_text = this_text.replace('-', '')
                    #this_text = this_text.replace('_', '')
                    tokens_t = tokenizer.tokenize(this_text)
                    input_token = torch.LongTensor(tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens_t + ['[SEP]']))
                    testData.append([input_token, segments_tensor[:len(input_token)], docID, this_text, sorted(com)])
        else:
            total_text = []
            for docid in args.docid:
                print(docid)
                tokens_tensor, segments_tensor, _, docID, origin_text = testData_raw[docid-1]
                total_text.extend(origin_text.split('.'))
                testData.append([tokens_tensor, segments_tensor, docID, origin_text, []])
            random.shuffle(total_text)
            text_split = ' '.join(total_text)
            text_split = np.asarray(text_split.split())
            if args.random:
                #comb = list(itertools.combinations(list(range(len(text_split))), args.N))
                r = list(range(len(text_split)))
                comb = [random.choices(r, k=args.N) for _ in range(args.random_num)]
            else:
                comb = [list(range(i, i+args.N)) for i in range(len(text_split)-args.N)]
            
            for com in comb:
                #print(tokens_tensor[com])
                this_text = ' '.join(text_split[sorted(com)])
                # Remove . & ,
                '''
                this_text = this_text.replace(',', '')
                this_text = this_text.replace('.', '')
                this_text = this_text.replace('(', '')
                this_text = this_text.replace(')', '')
                this_text = this_text.replace('[', '')
                this_text = this_text.replace(']', '')
                this_text = this_text.replace(':', '')
                this_text = this_text.replace(';', '')
                '''
                #this_text = this_text.replace('"', '')
                #this_text = this_text.replace('-', '')
                #this_text = this_text.replace('_', '')
                tokens_t = tokenizer.tokenize(this_text)
                input_token = torch.LongTensor(tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens_t + ['[SEP]']))
                testData.append([input_token, None, -1, this_text, sorted(com)])  

        model_data.append(testData)
        print(len(testData))
    logging.debug("END")

    if args.mode == 'train':
        logging.debug("Start training...")

        max_epoch = 10
        train(max_epoch, trainData, validData, device, model)

        logging.debug("END training")
    else:
        logging.debug("Start predicting...")
        
        best_query_lst = predict()

        logging.debug("END predicting")  

        logging.debug("Save query...")

        output_path = './results/bert_q_[SELECT_METHOD]_[N]_topk.txt'

        if args.random:
            output_path = output_path.replace('[SELECT_METHOD]', '{}_{}'.format("random", args.random_num))
        else:
            output_path = output_path.replace('[SELECT_METHOD]', 'inseq')

        output_path = output_path.replace('[N]', str(args.N))

        if not args.topk_rank:
            output_path = output_path.replace('_topk', '')
        
        with open(output_path, 'w') as f:
            for docID, query, _ in best_query_lst:
                f.write('{} {}\n'.format(docID, query))
        
        logging.debug("END")

        print("\n-----Here is the best query list preview-----")
        print(best_query_lst)
        print("\n-----Here is the best query list preview-----") 
          

    logging.debug("\n\nEND of program")
