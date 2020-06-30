import torch
import os
import json 
from tqdm import tqdm
import sys
import torch
from torch.utils.data import DataLoader

def train(max_epoch, trainData, validData, device, model):
    #model = BertForSequenceClassification.from_pretrained('bert-large-cased', hidden_dropout_prob=0.3, num_labels=NUM_LABELS)
    #history = {'train':[],'valid':[]}

    for epoch in range(max_epoch):
        print('Epoch: {}'.format(epoch))
        _run_epoch(epoch, True, trainData, validData, device, model)
        torch.cuda.empty_cache() 
        _run_epoch(epoch, False, trainData, validData, device, model)
        torch.cuda.empty_cache()

def _run_epoch(epoch, training, trainData, validData, device, model):
    model.train(training)
    if training:
        description = 'Train'
        dataset = trainData
        shuffle = True
    else:
        description = 'Valid'
        dataset = validData
        shuffle = False

    if training:
        BATCH_SIZE = 2
    else:
        BATCH_SIZE = 4
    dataloader = DataLoader(dataset, # 此處未修改到，需重train試驗結果
                          batch_size=BATCH_SIZE,
                          shuffle=shuffle, 
                          collate_fn=dataset.collate_fn,
                          num_workers=4)
    model.to(device)
    model.train(training)
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    criteria = torch.nn.BCELoss()
    history = {'train':[],'valid':[]}

    NUM_LABELS = 4

    trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=description, file=sys.stdout)
    loss = 0
    f1_score = F1()
    
    if training:
        for i, data in trange:
            tokens_tensors, segments_tensors, masks_tensors, labels, _= [t.to(device) for t in data]

            # 將參數梯度歸零

            # forward pass
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors
                            #, labels=labels
                            )
            # loss = outputs[0]
            # print(outputs[0], type(outputs[0]))

            pred_outputs = torch.sigmoid(outputs[0].view(-1, NUM_LABELS))
            # print(pred_outputs, pred_outputs.shape, labels, labels.shape)

            loss_fct = torch.nn.BCELoss()
            batch_loss = loss_fct(pred_outputs, labels)

            # batch_loss = criteria(outputs, labels)

            if training:
                opt.zero_grad()
                # backward
                batch_loss.backward()

                # 梯度下降過慢可嘗試以下gradient clip metho
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)

                opt.step()
                # add learning rate scheduler
                #scheduler.step()

            # 紀錄當前 batch loss
            loss += batch_loss.item()
            #print(pred_outputs.cpu())
            #print(data[3])
            f1_score.update(pred_outputs.cpu(), data[3]) # argument 1要放有沒有sigmoid過的有待驗證
            #print("", end='')

            trange.set_postfix(
                loss=loss / (i + 1), f1=f1_score.print_score())
            '''
            可能問題: 前三個都未預測出的label應填 'others" 
            '''

            '''
            o_labels, batch_loss = _run_iter(x,y)
            if training:
                opt.zero_grad()
                batch_loss.backward()
                opt.step()

            loss += batch_loss.item()
            f1_score.update(o_labels.cpu(), y)

            trange.set_postfix(
                loss=loss / (i + 1), f1=f1_score.print_score())
            '''
    else:
        with torch.no_grad():
            for i, data in trange:
                tokens_tensors, segments_tensors, masks_tensors, labels, _ = [t.to(device) for t in data]

                # 將參數梯度歸零

                # forward pass
                outputs = model(input_ids=tokens_tensors, 
                                token_type_ids=segments_tensors, 
                                attention_mask=masks_tensors
                                #, labels=labels
                                )
                # loss = outputs[0]
                # print(outputs[0], type(outputs[0]))

                pred_outputs = torch.sigmoid(outputs[0].view(-1, NUM_LABELS))
                # print(pred_outputs, pred_outputs.shape, labels, labels.shape)

                loss_fct = torch.nn.BCELoss()
                batch_loss = loss_fct(pred_outputs, labels)

                # batch_loss = criteria(outputs, labels)


                # 紀錄當前 batch loss
                loss += batch_loss.item()
                #print(pred_outputs.cpu())
                #print(data[3])
                f1_score.update(pred_outputs.cpu(), data[3]) # argument 1要放有沒有sigmoid過的有待驗證
                #print("", end='')

                trange.set_postfix(
                    loss=loss / (i + 1), f1=f1_score.print_score())
                '''
                可能問題: 前三個都未預測出的label應填 'others" 
                '''

                '''
                o_labels, batch_loss = _run_iter(x,y)
                if training:
                    opt.zero_grad()
                    batch_loss.backward()
                    opt.step()

                loss += batch_loss.item()
                f1_score.update(o_labels.cpu(), y)

                trange.set_postfix(
                    loss=loss / (i + 1), f1=f1_score.print_score())
                '''

    if training:
        history['train'].append({'f1':f1_score.get_score(), 'loss':loss/ len(trange)})
    else:
        history['valid'].append({'f1':f1_score.get_score(), 'loss':loss/ len(trange)})

    save(epoch, model, history, init_epoch=0)


def save(epoch, model, history, init_epoch=0):
    # save to colab local
    torch.save(model.state_dict(), './large_model_test.pkl.'+str(epoch+init_epoch))
    with open('./large_test_history.json', 'w') as f:
        json.dump(history, f, indent=4)

class F1():
    def __init__(self, threshold=0.4):
        self.threshold = threshold # original: 0.5, 70%: 0.4, 0.3~0.35最好(統一時)
        self.n_precision = 0
        self.n_recall = 0
        self.n_corrects = 0
        self.name = 'F1'

    def reset(self):
        self.n_precision = 0
        self.n_recall = 0
        self.n_corrects = 0

    def update(self, predicts, groundTruth):
        predicts = predicts > self.threshold
        '''
        newpredicts = []
        #print(predicts)
        for predict in predicts.data.tolist():
            #print(predict)
            if predict == [0, 0, 0, 0]: # OTHERS
                newpredicts.append([0, 0, 0, 1])
            else:
                predict[3] = 0 # OTHERS for 0
                newpredicts.append(predict)
        predicts = torch.tensor(newpredicts)
        #print(predicts)
        '''
        self.n_precision += torch.sum(predicts).data.item()
        self.n_recall += torch.sum(groundTruth).data.item()
        # print(groundTruth.type(torch.uint8)*predicts.type(torch.uint8), (groundTruth.type(torch.uint8)*predicts.type(torch.uint8)).dtype)
        # print(predicts.type(torch.uint8))
        self.n_corrects += torch.sum(groundTruth.type(torch.uint8) * predicts.type(torch.uint8)).data.item()

    def get_score(self):
        recall = self.n_corrects / self.n_recall
        precision = self.n_corrects / (self.n_precision + 1e-20)
        return 2 * (recall * precision) / (recall + precision + 1e-20)

    def print_score(self):
        score = self.get_score()
        return '{:.5f}'.format(score)
