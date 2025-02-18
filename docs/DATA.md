



In the following, we discuss
1. [Log Files Corpus](#logs)
2. [Log Files Embeddings](#embeddings)
3. [Submission Format](#sub-format)
4. [Metadata](#metadata)

---
#### Log Files Corpus <a name="logs"></a>

The full curated dataset of log files can be found [here in log_files.zip](https://analog-my.sharepoint.com/:u:/p/ash_aldujaili/EdbF7ipF0C5Gs_UTs3MchzUBz1p3-O4t0xsReyU4p2qwOQ?e=4JY03a). 

It has the following structure:
```
.
├── log_1.log // one log file generated by one test
├── log_2.log 
.....  
└── log_792.log 
```
Please unzip these log files to be at `$REPO_PATH/data/log_files`

We thank Kaushal Modi for sharing this corpus of data with us.

---
#### Log Files Embeddings <a name="embeddings"></a>

One approach to identify similarities among log files is to:

1. Embed them into an embedding space (see [here](https://platform.openai.com/docs/guides/embeddings) for more details on embeddings). Now, each log file is represented by a list of floating point numbers.
2. Use/learn/Design a similarity metric/function that quantifies how similar two log files are given their embeddings (e.g., Euclidean distance).
3. Employ the similarity values obtained from (2.) to cluster (e.g., using k-means) the log files into clusters that ideally would correspond to common root causes 

There are several ways to go about (1.) either because of its technical challenges or the nature of the text to be embedded (see [here](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/) for more details embedding textual data)


To ensure the data used for this challenge does *not* get shared publicly and to make this challenge accessible to particiapnts across ADI who might not have the resources to run large embedding models, we make use of the `fastembed` python package to embed these log files on one's local machine (no GPU is required, but having a GPU certainly makes the embedding process faster). Besides protecting ADI's data (by using embedding models locally), this gives the chance to participants to experiment with the chunking/embedding strategies.

We provide an example of embedding log files with `fastembed` with a simple text chunking strategy in `scripts/embed_mlchallenge_data.py`
After running this script, its output directory will have the following structure:
```
.
├── log_1.npy // one embedding numpy file per log file (an array of mxn)
├── log_2.npy 
.....  
└── log_792.npy 
```

To save you the time of regenerating these numpy files, you can download them from [log_embeddings.zip](https://analog-my.sharepoint.com/:u:/p/ash_aldujaili/EfA_u9NoZZJArPab-8vNE5IBeLOrs4jXTJY-T21LuY-1Mw?e=dZqgs3)
Please unzip these log files to be at `$REPO_PATH/data/log_embeddings`.

Note again that these embeddings do not necessarily represent the best embeddings one can obtain given the different embedding models and chunking strategies out there. Consider experimenting with that.


---
#### Submission Format <a name="sub-format"></a>

For evaluation/ranking purposes, we suggest participants send us their entries in a CSV file with the following format:
(`fname` column has str entries, `label` column has int entries >=0)

```
fname, label
log_1.log, LABEL_OF_LOG_1
log_2.log, LABEL_OF_LOG_2
...
log_792.log, LABEL_OF_LOG_729
```


For convenience, we provide a helper Python function to write the submission into a CSV file. Check [this](https://gitlab.analog.com/aaldujai/ml_challenge_as24/-/blob/master/docs/EVAL.md).


---
#### Other Metadata <a name="metadata"></a>

Besides the log files and their embeddings, we provide a noisy clustering of some of these log files in `$REPO_PATH/data/log_metadata/mlchallenge_labels.csv`.
This file takes the same format as that of a submission file, with the labels of some of the files being -1 to denote a missing label (around 1/4th of the files).

Note that, besides the -1 missing labels, this file does not enumerate all the log files present in `data/log_files` 




