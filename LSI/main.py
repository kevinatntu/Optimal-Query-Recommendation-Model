import vsm
"""
queries
    - list
    - 可以多個query
    - 字中間以空格隔開，會處理標點符號
scores_follow_filelist 
    - 每一個document 的分數，順序是依照當前檔案的file list
descending_ranking_by_id
    - 排名，數字代表當前檔案的file list 的第幾個檔案
    - 逆序，排在第一位代表分數最高第一名
"""
import json
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser(description="Here is your hyperparameter settings")
parser.add_argument('--dim', type=int, default=100, help='SVD latent dimension')
parser.add_argument('--qlen', type=int, default=5, help='length of query to calculate rank1 percentage')
parser.add_argument('--tid', type=int, default=2, help='target document id to generate query to calculate mAP, ranges from 0 to 4')
parser.add_argument('--model_dir', type=str, default='./model/', help='directory of inverted file')
args = parser.parse_args()


# dim = 5000
print('---------------')
print('dim=', args.dim)
print('qlen=', args.qlen)
print('target doc id', args.tid)
print('---------------')
v = vsm.VSM(model_dir=args.model_dir, dim=args.dim)

# with open('test.json', 'w') as f:
#     json.dump(v.TF, f)
# with open('vocab.json', 'w') as f:
#     json.dump(v.vocab, f)
# print(v.vocab.shape)
# print(v.vocab[1000])
# print(v.filelist[1349])
import numpy as np


def calc_AP(ret, ans):
    AP = 0
    hit = 0
    for i, v in enumerate(ret):
        if v in ans:
            hit += 1
            AP += hit / (i + 1)
    AP /= len(ans)
    #  if flag and AP!=0.0:
        #  print(ret,ans)
        #  print('AP',AP)
    return AP

with open('mini-file-list', 'r') as f:
# with open('./model/file-list', 'r') as f:
    files = f.readlines()
files = [f.strip() for f in files]

file2id = { f:int(f[1:])-1 for f in files }

target_doc = ['D00076', 'D01032', 'D01350', 'D02582', 'D05005']
target_id = [ file2id[t] for t in target_doc]

# mean_query = []
smooth_tfidf = v.approx[:,target_id[args.tid]]
idx = np.argsort(smooth_tfidf)[::-1]
queries = [ ' '.join([v.vocab[i] for i in idx[:10]]), ' '.join([v.vocab[i] for i in idx[:5]]), 
            ' '.join([v.vocab[i] for i in idx[:1]])]
# for tid in target_id:
#     smooth_tfidf = v.approx[:,tid]
#     idx = np.argsort(smooth_tfidf)[::-1][0]
#     print('file id %d, words: %s' %(tid, v.vocab[idx]))
#     mean_query.append(v.vocab[idx])
# mean_query = ' '.join(mean_query)
scores_follow_filelist, descending_ranking_by_id = v.retrieval(queries)
# scores_follow_filelist, descending_ranking_by_id = v.retrieval([mean_query])
AP = calc_AP(descending_ranking_by_id[0].tolist(), target_id)
print('10 words query',AP, queries[0])
AP = calc_AP(descending_ranking_by_id[1].tolist(), target_id)
print('5 words query',AP, queries[1])
AP = calc_AP(descending_ranking_by_id[2].tolist(), target_id)
print('1 words query',AP, queries[2])

# qlen = 5
ranking_list = []
for file in tqdm(files):
    # print('file:', file)
    # print('file_id', file2id[file])
    file_id = file2id[file]
    smooth_tfidf = v.approx[:,file_id]
# print(np.sort())
    idx = np.argsort(smooth_tfidf)[::-1]

    # for i in idx[:qlen]:
        # print(v.vocab[i])
# exit()
# queries = [ ' '.join([v.vocab[i] for i in idx[:10]]), ' '.join([v.vocab[i] for i in idx[:5]])]
    queries = [ ' '.join([v.vocab[i] for i in idx[:args.qlen]])]
# queries = ['object recognition detection neural YOLO']
# queries = ['YOLO']
    # print(queries[0])
# for example:
# queries = ["Adversarial Examples that Fool Detectors"]
    scores_follow_filelist, descending_ranking_by_id = v.retrieval(queries)
    # print(file_id)
    rank = descending_ranking_by_id[0].tolist().index(file_id) + 1
    ranking_list.append(rank)
print(ranking_list)
# print(files)



ranking_list = np.array(ranking_list)
top1 = len(ranking_list[ranking_list<=1])
top5 = len(ranking_list[ranking_list<=5])
top10 = len(ranking_list[ranking_list<=10])
print('top1=%d, top5=%d, top10=%d'%(top1, top5, top10))
import matplotlib.pyplot as plt
plt.hist(ranking_list)
plt.xlabel('Ranking')
plt.ylabel('Numbers')
plt.title('Ranking Distribution (dim=%d)' %(args.dim))
plt.show()

# with open('ranking_list.txt', 'w') as f:
#     for i, (s, idx) in enumerate(zip(scores_follow_filelist[0], descending_ranking_by_id[0])):
#         # if i >100: break
#         # if str(v.filelist[idx]) in files:
#         f.write(str(s)+' '+str(v.filelist[idx])+'\n')