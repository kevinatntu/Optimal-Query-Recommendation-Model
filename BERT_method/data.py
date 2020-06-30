from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import re

from transformers import *

class BERTDataset(Dataset):
    def __init__(self, mode, datapath, tokenizer, pad_idx, max_len = 500):
        self.data = pd.read_csv(datapath, dtype=str)
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.mode = mode
        assert mode in ['train', 'test']
        self.label_map = {'THEORETICAL': 0, 'ENGINEERING':1, 'EMPIRICAL':2, 'OTHERS':3}
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # text_list = []
        if self.mode == "test": # select from original data
            text_list = self.data.iloc[index, 2].split('$$$')
            # text_list = self.data.iloc[index, 1]
            label_tensor = None
        else:
            text_list = self.data.iloc[index, 1].split('$$$') # select from trainset.csv
            labels = self.data.iloc[index, 2].split(' ')
            # 將 label 文字也轉換成索引方便轉換成 tensor
            onehot = [0,0,0,0]
            for l in labels:
                onehot[self.label_map[l]] = 1
            label_tensor = torch.FloatTensor(onehot)
        
        docID = re.search('D([0-9]+)', self.data.iloc[index, 0]).group(1)
        docID = int(docID)
        
        word_pieces = []
        seg_pieces = []
        prev_len = 0
        for text_idx, text in enumerate(text_list):    
            if text_idx == 0:
              word_pieces += ["[CLS]"]
            
            tokens_t = self.tokenizer.tokenize(text)

            # try to limit the maxlen of one document('Abstract')
            #if prev_len + len(tokens_t) > 500:
            #  break
            
            # word_pieces += tokens_t + ["[SEP]"]
            word_pieces += tokens_t 
            
            len_t = len(word_pieces) - prev_len
            prev_len = len(word_pieces)
        
        if prev_len > 500: # 目前改為400降低長度
            word_pieces = word_pieces[:500]

        word_pieces += ["[SEP]"]
            
        seg_pieces = [0] * len(word_pieces) # 0 or 1? good question
        
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.LongTensor(ids)
            
        segments_tensor = torch.LongTensor(seg_pieces)    

        origin_text = " ".join(text_list)
        
        return (tokens_tensor, segments_tensor, label_tensor, docID, origin_text)
    def collate_fn(self, samples):
        tokens_tensors = [s[0] for s in samples]
        segments_tensors = [s[1] for s in samples]
        
        # 測試集有 labels
        if samples[0][2] is not None:
            label_ids = torch.stack([s[2] for s in samples])
        else:
            label_ids = None
        
        # zero pad 到同一序列長度
        tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
        segments_tensors = pad_sequence(segments_tensors, batch_first=True)
        
        # attention masks，將 tokens_tensors 裡頭不為 zero padding
        # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
        masks_tensors = torch.zeros(tokens_tensors.shape, 
                                    dtype=torch.long)
        masks_tensors = masks_tensors.masked_fill(
            tokens_tensors != 0, 1)

        #docID_lst = [s[3] for s in samples]
        text_lst = [s[4] for s in samples]
        
        return tokens_tensors, segments_tensors, masks_tensors, label_ids, text_lst

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
    validData = BERTDataset("train", '../dataset/validset.csv', tokenizer, 0)
    testData = BERTDataset("test", '../dataset/task2_trainset.csv', tokenizer, 0)

    print("DOCID: ", validData[5][3])
    print("DOCID: ", testData[1350-1][3])
    print(testData[1350-1][4])

    BATCH_SIZE = 4
    trainloader = DataLoader(testData, batch_size=BATCH_SIZE, 
                            collate_fn=validData.collate_fn)

    # test correctness
    data = next(iter(trainloader))

    tokens_tensors, segments_tensors, \
        masks_tensors, label_ids, text_lst = data

    print(f"""
    tokens_tensors.shape   = {tokens_tensors.shape} 
    {tokens_tensors}
    ------------------------
    segments_tensors.shape = {segments_tensors.shape}
    {segments_tensors}
    ------------------------
    masks_tensors.shape    = {masks_tensors.shape}
    {masks_tensors}
    ------------------------
    {text_lst}
    """)