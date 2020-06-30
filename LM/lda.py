import os
import numpy as np
from gensim.models.ldamodel import LdaModel
from scipy.stats import dirichlet

class LDA():
    def __init__(self, corpus_dir, lda_weight=0.5):
        self.corpus_dir = corpus_dir
        with open(os.path.join(self.corpus_dir, "file-list"), 'r') as f:
            self.id_file = dict(enumerate([i.strip() for i in f.readlines()]))
            self.file_id = dict(map(reversed, self.id_file.items()))
        with open(os.path.join(self.corpus_dir, "vocab.all"), 'r') as f:
            self.id_vocab = dict(enumerate([v.strip() for v in f.readlines()]))
            self.vocab_id = dict(map(reversed, self.id_vocab.items()))
        self.tf = self.create_tf()
        self.corpus = self.create_corpus(self.tf)
        lda_path = os.path.join(corpus_dir, "lda.pkl")
        self.lda = LdaModel.load(lda_path) if os.path.exists(lda_path) else self.create_lda(lda_path)
        self.lda_weight = lda_weight
        self.lm = self.create_lm()


    def create_tf(self):
        print("\nBuilding corpus...", flush=True)
        tf = np.zeros((len(self.id_vocab), len(self.id_file)))
        with open(os.path.join(self.corpus_dir, "inverted-file"), 'r') as f:
            inverted_file = f.readlines()
        i = 0
        while i < len(inverted_file):
            v = [int(j) for j in inverted_file[i].split()]
            assert len(v) == 3
            for j in range(v[2]):
                i += 1
                d = [int(j) for j in inverted_file[i].split()]
                tf[v[0]][d[0]] = d[1]
            i += 1
        df = np.sum(np.where(tf > 0, 1, 0), axis=1).reshape(-1, 1)        
        hfword_id = np.argsort(-df.flatten())[:50]
        tf[hfword_id] = 0
        for i in hfword_id:
            print(self.id_vocab[i])
         
        return tf

    def create_corpus(self, tf):
        corpus = [[] for i in range(tf.shape[1])]
        nonzero_id = np.nonzero(tf)
        for t, d in zip(nonzero_id[0].tolist(), nonzero_id[1].tolist()):
            corpus[d].append((t, tf[t][d]))

        return corpus


    def create_lda(self, lda_path):
        print("\nBuilding LDA model...", flush=True)
        lda = LdaModel(self.corpus, num_topics=32, id2word=self.id_vocab, iterations=int(1e+7), chunksize=1024, update_every=2, passes=4, alpha=5e-4, random_state=14, minimum_probability=1e-4)
        #lda.save(lda_path)
        
        return lda

    def create_lm(self):
        #target_id = self.file_id["D01350"]
        #print(self.lda[self.corpus[target_id]])
        #for i in self.lda[self.corpus[target_id]]:
        #    print(i)
        #    print(self.lda.print_topic(i[0], 20))
        #print(np.mean([i[1] for i in self.lda.top_topics(self.corpus, topn=64)]))
        
        print("\nBuilding prob matrix...", flush=True)
        vocab_topic = self.lda.get_topics().T
        topic_doc = np.zeros((vocab_topic.shape[1], self.tf.shape[1]))
        for i, j in enumerate(self.lda.get_document_topics(self.corpus)):
            for k in j:
                topic_doc[k[0]][i] = k[1]

        #smoothed_lm = (self.tf + 10 * np.sum(self.tf, axis=1).reshape(-1, 1) / np.sum(self.tf)) / (np.sum(self.tf, axis=0).reshape(1, -1) + 10)
        mle_lm = self.tf / np.sum(self.tf, axis=0).reshape(1, -1)
        lda_lm = np.dot(vocab_topic, topic_doc)
        lda_lm = lda_lm / np.sum(lda_lm, axis=0).reshape(1, -1)
        
        return np.log(1e-10 + (1 - self.lda_weight) * mle_lm + self.lda_weight * lda_lm)
    

    def retrieve_single(self, target_file, N):
        print("\nRetrieving {}...".format(target_file), flush=True)
        target_id = self.file_id[target_file]
        term_rank = np.argsort(-self.lm[:, target_id])
        for i in term_rank[:10]:
            print(self.id_vocab[i], self.tf[i][target_id], np.exp(self.lm[i][target_id]))
        
        for i in N:
            q = np.zeros((self.lm.shape[0], 1))
            q[term_rank[:i]] = 1
            rel = np.dot(self.lm.T, q).flatten()
            rel_ids = np.argsort(-rel)
            #for j in ["D00076", "D01032", "D01350", "D02582", "D05005"]:
            #    print(np.where(rel_ids == self.file_id[j])[0][0] + 1)
            print(np.where(rel_ids == target_id)[0][0] + 1)
   

    def retrieve_multiple(self, target_files, N):
        print("\nRetrieving {}...".format(target_files), flush=True)
        target_ids = [self.file_id[f] for f in target_files]
        tf = np.sum(self.tf[:, target_ids], axis=1).reshape(-1, 1)
        corpus = self.create_corpus(tf)
        vocab_topic = self.lda.get_topics().T
        topic_doc = np.zeros((vocab_topic.shape[1], 1))
        for k in self.lda.get_document_topics(corpus)[0]:
            topic_doc[k[0]][0] = k[1]
        mle_lm = tf / np.sum(tf)
        #smoothed_lm = (tf + 10 * np.sum(self.tf, axis=1).reshape(-1, 1) / np.sum(self.tf)) / (np.sum(tf) + 10)
        lda_lm = np.dot(vocab_topic, topic_doc)
        lm = np.log(1e-10 + (1 - self.lda_weight) * mle_lm + self.lda_weight * lda_lm)
        
        term_rank = np.argsort(-lm[:, 0])
        for i in term_rank[:10]:
            print(self.id_vocab[i], tf[i][0], np.exp(lm[i][0]))
        
        for i in N:
            q = np.zeros((self.lm.shape[0], 1))
            q[term_rank[:i]] = 1
            rel = np.dot(self.lm.T, q).flatten()
            rel_ids = np.argsort(-rel)
            for target_id in target_ids:
                print(i, self.id_file[target_id], np.where(rel_ids == target_id)[0][0] + 1, np.exp(rel[target_id]))
