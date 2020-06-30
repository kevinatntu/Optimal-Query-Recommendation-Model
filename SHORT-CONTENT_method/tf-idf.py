#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import operator


# In[2]:


import vsm
corpus = 'corpus/model'
target_id = 'D01350'

all_target_id = ['D00076', 'D01032', 'D01350', 'D02582', 'D05005']#.remove(target_id)


# In[11]:


def cal_mAP(lt_gen, lt_ground_truth):
    MAP = 0
    a = 0
    correct = 0
    precision = []
    print(lt_gen)
    for i in lt_gen:
        a+=1
        if i in lt_ground_truth:
            correct+=1
            print('c', correct, 'a', a)
            precision.append(correct/a)
    MAP=sum(precision)/len(lt_ground_truth)
    return MAP


# In[12]:


v = vsm.VSM(model_dir=corpus)


# ## Stage 1

# In[13]:


df_whole = pd.read_csv('create_corpus/task2_trainset.csv')
df_whole = df_whole.set_index('Id')
target = df_whole.loc[target_id]
target_whole_query = target['Abstract']
target_whole_query = re.sub(r'[^\w\s]',' ',target_whole_query).lower()#.split(' ')
target_whole_query =[target_whole_query]


# In[14]:


scores_follow_filelist, descending_ranking_by_id = v.retrieval(target_whole_query)


# In[17]:


cal_mAP(v.filelist[descending_ranking_by_id[0]], all_target_id)


# In[19]:


target_pos_in_file_list = descending_ranking_by_id[0][0]


# ## Stage 2

# In[20]:


def get_all_unique_words(df):
    all_words = []
    for ab in tqdm(df['Abstract']):

        abb = re.sub(r'[^\w\s]',' ',ab).lower().split(' ')
        abbb = np.unique(abb)#, return_counts=True)
        all_words += list(abbb)
    return all_words

def create_vocab_all(all_words):
    lstt_word = list(np.unique(all_words))
    return lstt_word

def create_inverted_file(df, lstt_word):
    inverted_file = list([] for i in range(len(lstt_word)))
    for index, ab in tqdm(enumerate(df['Abstract']), total=len(df['Abstract'])):
        abb = re.sub(r'[^\w\s]',' ',ab).lower().split(' ')
        w,c = np.unique(abb, return_counts=True)
        for _w, _c in zip(w,c):
            idx = lstt_word.index(_w)
            inverted_file[idx].append([index, _c])
    return inverted_file


# In[21]:


df_corpus = df_whole.loc[v.filelist]
df_corpus = df_corpus.reset_index()


# In[22]:


all_words = get_all_unique_words(df_corpus)
lstt_word = create_vocab_all(all_words)
inverted_file = create_inverted_file(df_corpus, lstt_word)


# ### Exclude stop words
# * Top 100 highest frequency words

# In[23]:


## find out the frequency of all words
freq_vocab = []
for doc in inverted_file:
    count = 0
    for i in range(len(doc)):
        count+=doc[i][1]
    freq_vocab.append(count)


# In[24]:


## sort the words by its frequency in descending order
## freq = index of words 
freq = np.argsort(freq_vocab)[::-1]


# In[94]:


import matplotlib.pyplot as plt
# plt.rcdefaults()
fig, ax = plt.subplots()
top =20
# Example data
people = stop_words[1:top]
performance = np.array(freq_vocab)[freq[1:top]]
y_pos = np.arange(len(people))

ax.barh(y_pos, performance)
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Frequency of appearing in corpus')
ax.set_title('Top 20 frequent words(stop words)in corpus')

plt.show()


# In[60]:


## find out the Top 100 highest frequency words as the stop words
stop_words = np.array(lstt_word)[freq[:101]]


# In[27]:


## exclude the stop words from the target abstract 
target_no_stop_query = ''
for i in target_whole_query[0].strip('').split(' '):
    if i!='' and i not in stop_words:
        target_no_stop_query+=i
        target_no_stop_query+=' '
    # target_whole_query


# In[28]:


target_no_stop_query = [target_no_stop_query ]


# In[29]:


scores_follow_filelist, descending_ranking_by_id = v.retrieval(target_no_stop_query)


# In[31]:


cal_mAP(v.filelist[descending_ranking_by_id[0]], all_target_id)


# ### No repeat words

# In[32]:


w,c = np.unique(target_no_stop_query[0].split(' '), return_counts=True)


# In[33]:


repeated_word = []
for (_w,_c) in zip(w,c):
    if _c >1:
        repeated_word.append(_w)


# In[34]:


## exclude the stop words from the target abstract 
target_no_repeat_query = ''
tmp_query = []
for i in target_no_stop_query[0].strip('').split(' '):
    if i!='' and i not in tmp_query:
        target_no_repeat_query+=i
        tmp_query.append(i)
        target_no_repeat_query+=' '
target_no_repeat_query


# In[35]:


target_no_repeat_query= [target_no_repeat_query]


# In[36]:


scores_follow_filelist, descending_ranking_by_id = v.retrieval(target_no_repeat_query)


# In[38]:


cal_mAP(v.filelist[descending_ranking_by_id[0]], all_target_id)


# ### Decreasing number of words
# * by tf-idf
# 

# In[39]:



dict_tfidf = {}
for term in target_no_repeat_query[0].lower().split(' '):
    if term != '':
        r = v.TF[term][target_pos_in_file_list] * v.IDF[term]
        dict_tfidf[term] = r
dict_tfidf = sorted(dict_tfidf.items(), key=operator.itemgetter(1))


# In[109]:


# dict_tfidf
all_true = []
query_remove_by_ifidf = target_no_repeat_query[0].lower().split(' ')
with open('query.txt', 'w') as fp:
    
    for term, y in dict_tfidf:
        query_remove_by_ifidf.remove(term)
        target_query_remove_by_ifidf = ''
        for i in query_remove_by_ifidf:
            if i != '':
                target_query_remove_by_ifidf += i
                target_query_remove_by_ifidf += ' '

        if target_query_remove_by_ifidf == '':
            break

#         if len(query_remove_by_ifidf) ==4:
#             break
        print('target_query_remove_by_ifidf', target_query_remove_by_ifidf)
        fp.writelines(target_query_remove_by_ifidf+'\n')
        target_query_remove_by_ifidf = [target_query_remove_by_ifidf]
        scores_follow_filelist, descending_ranking_by_id = v.retrieval(target_query_remove_by_ifidf)    
        print(v.filelist[descending_ranking_by_id[0][0]] == target_id    )
        if not v.filelist[descending_ranking_by_id[0][0]] == target_id :
            print(descending_ranking_by_id)
        all_true.append(v.filelist[descending_ranking_by_id[0][0]] == target_id)

    t = True
    for i in all_true:
        t*=i
    print(t==1)


# In[110]:



cal_mAP(v.filelist[descending_ranking_by_id[0]], all_target_id)

