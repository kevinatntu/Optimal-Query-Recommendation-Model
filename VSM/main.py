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

v = vsm.VSM(model_dir='mini_corpus_1')

queries = ["1st query", "2nd query"]
# for example:
# queries = ["Adversarial Examples that Fool Detectors"]
scores_follow_filelist, descending_ranking_by_id = v.retrieval(queries)