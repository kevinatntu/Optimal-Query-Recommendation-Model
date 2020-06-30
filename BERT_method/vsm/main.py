import vsm
import argparse
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

#from pathlib import Path
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

all_doc_lst = [76, 1032, 1350, 2582, 5005]

def _parse_args():
    parser = argparse.ArgumentParser(
        description="VSM"
    )
    parser.add_argument('--doc_path', type=str, help='Corpus folder', nargs='+', 
    default=['../mini_corpus/mini_corpus_1','../mini_corpus/mini_corpus_2','../mini_corpus/mini_corpus_3','../mini_corpus/mini_corpus_4'
    ,'../mini_corpus/mini_corpus_5','../mini_corpus/full_corpus']) 
    parser.add_argument('--query_paths', type=str, help="Query file path", nargs='+', default=[])
    parser.add_argument('--results_path', type=str, help='Query folder', default='../results/') 
    parser.add_argument('--topk', action="store_true", help='All do top-k ranking')
    parser.add_argument('--plot', action="store_true", help='Plot figure')
    
    args = parser.parse_args()
    return args

def calculate_scores(ranking, doc, docID_list):
    if doc != 'Top k': # one docid at a time
        doc = int(doc)
        doc_idx = docID_list.index(doc)
        rank = ranking.index(doc_idx) + 1
        top10_lst = [docID_list[r] for r in ranking[:10]]
        return rank, top10_lst
    else: # map
        #dlst = np.where(docID_list == all_doc_lst)
        dlst = [docID_list.index(d) for d in all_doc_lst]
        rs = [ranking.index(r) for r in dlst]
        map_scores = 0
        #rs.sort()
        for i, n in enumerate(sorted(rs)):
            map_scores += (i+1) / (n+1)
        map_scores /= len(all_doc_lst)
        return map_scores, rs

if __name__ == '__main__':
    args = _parse_args()

    if not args.query_paths:
        if not os.path.exists(args.results_path):
            print("Neither query nor folder found!")
            exit(1)
        args.query_paths = glob.glob(args.results_path + "*.txt")

    print(args.query_paths)

    for doc_p in args.doc_path:
        print("Init VSM...")
        v = vsm.VSM(model_dir=doc_p)

        print("Calculate ranking...")
        query_lst = []
        query_docid_lst = []
        query_name_lst = []
        queries = []
        for filename in args.query_paths:
            with open(filename, 'r') as f:
                this_query_lst = f.readlines()
                for q in this_query_lst:
                    q = q.strip().split(' ') # [docID, query_text]
                    docID, query = int(q[0]), ' '.join(q[1:])
                    queries.append(query)
                    #print(docID, query)
                    if docID == -1 or args.topk:
                        query_docid_lst.append("Top k")
                    else:
                        query_docid_lst.append(docID)
                    query_name_lst.append(filename)    

        #queries = ["1st query", "2nd query"]
        # for example:
        # queries = ["Adversarial Examples that Fool Detectors"]
        scores_follow_filelist, descending_ranking_by_id, docID_list = v.retrieval(queries)

        print("Show ranking...\n")
        results = ''

        ranks = []

        for idx, ranking in enumerate(descending_ranking_by_id):
            cur_results = ''
            cur_results += "For file [{}], query [{}]\n".format(query_name_lst[idx], queries[idx])
            cur_results += 'DocID: {}\n'.format(query_docid_lst[idx])
            item1, item2 = calculate_scores(ranking.tolist(), query_docid_lst[idx], docID_list.tolist())
            ranks.append(item1)
            if query_docid_lst[idx] != 'Top k':
                cur_results += "Among {} documents, rank {}\nQuery's top 10: {}\n\n".format(
                    len(docID_list), item1, item2)
            else:
                cur_results += "Among {} documents, MAP = {}\nEach ranking: {}\n\n".format(
                    len(docID_list), item1, item2)
            
            print(cur_results)
            results += cur_results

        corpus_name = doc_p[doc_p.rfind('/')+1:]
        with open('./output_{}.txt'.format(corpus_name), 'w') as f:
            f.write(results)
        
        if args.plot:
            ranks = np.asarray(ranks)
            top1_ranks = ranks[ranks == 1]
            top5_ranks = ranks[ranks <= 5]

            print("About {}% rank 1".format(100 * len(top1_ranks) / len(ranks)))
            print("About {}% rank top 5".format(100 * len(top5_ranks) / len(ranks)))

            # plot distribution
            plt.figure(figsize=(10, 8))
            plt.hist(ranks)
            plt.title("Ranking distribution - In-sequence selected")
            plt.xlabel("Rank")
            plt.ylabel("Number of documents")
            plt.savefig('inseq_distribution.png')
            plt.show()