# BERT - learning-based model

## main folder
- For BERT model training / Query generating
  - Training
  > python main.py --train --model_save_path=[Model saving path]
  
  - Query generating
  > python main.py --test --docid [Target DocID] --N=[Max #words in query] --random --random_num=[#sampling] 
    * Options
      - docid: Target DocID that you want to generate optimal query for. Support multiple inputs. If not specified, tool will automatically randomly select 100 documents from corpus.
      - N: Number of words in the optimal query
      - topk_rank: If specified, turn on the top-k mode for the set of DocIDs
      - random_num: Number of sampling from original document. For example, if 1000, tool will generate 1000 query candidates from the documents, then calculate their MSE and select the minimum one as optimal query

## vsm folder
- Special version, for BERT's query only
  - Command
  > python ./vsm/main.py --doc_path [Target corpus folder path] --query_paths [Optimal query path]
    * Options
      - doc_path: Corpus folder. Support multiple inputs. Tool will use your optimal query to rank all DocID in the specified corpus
      - query_paths: Optimal Query input path. Support multiple inputs
      - results_path: If specified, then use all query files in this path to do ranking. You can directly put all optimal query files in one folder, then pass it as results_path's argument
      - topk: If specified, do top-k mAP task regardless of your optimal query
      - plot: If specified, plot the distribution of ranking
      
      
      
