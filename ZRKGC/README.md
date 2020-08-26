# \#ZRKGC

 

## \###Dependency

The code has been tested with:

\* python 3.6 

\* pytorch 1.1.0

\* Ubuntu 18.04



You first need to create an environment using `anaconda` or `virtualenv` and then activate the env.

Please follow  (https://github.com/microsoft/unilm/tree/master/unilm-v1) to build the unilm environment, especially apex and nlg-eval.

Other dependencies need

```
pip install -r requirements.txt
```

## \###Model

Download from (https://github.com/microsoft/unilm/tree/master/s2s-ft) to get `[unilm1.2-base-uncased]` model. The model are trained by using the same model configuration and WordPiece vocabulary as BERT Base.

Note that  `unilm_v2_bert_pretrain` folder shoud  contains  three components: `bert_config.json` 、`unilm1.2-base-uncased.bin`  and  `vocab.txt`.



## \###Data

### Train data

Download and unzip from Google Drive (https://drive.google.com/file/d/1bKjHtJMDwxsXRwQD2UQg37TdEfVg6l3k/view?usp=sharing).  Because of the large size of train data , this process will take some time.

### Test data

#### Wizard of Wikipedia 

Download Wizard of Wikipedia dataset from (https://parl.ai/projects/wizard_of_wikipedia/), then preprocess it by 

```
python preprocess/wizard_preprocess.py ${data_path}/test_random_split.json

python preprocess/wizard_preprocess.py ${data_path}/test_topic_split.json
```



#### Topical Chat

Topical Chat dataset can be downloaded from (https://github.com/alexa/alexa-prize-topical-chat-dataset).

Follow the instructions to set  environment and build data. Then preprocess the builded data by 

```
python3 preprocess/topical_preprocess.py ${data_path}/alexa-prize-topical-chat-dataset/conversations/test_freq.json ${data_path}/alexa-prize-topical-chat-dataset/reading_sets/post-build/test_freq.json

python3 preprocess/topical_preprocess.py ${data_path}/alexa-prize-topical-chat-dataset/conversations/test_rare.json ${data_path}/alexa-prize-topical-chat-dataset/reading_sets/post-build/test_rare.json
```



#### CMU_DoG

Download CMU_DoG dataset fom (https://github.com/lizekang/ITDD), then preprocess it by

```
python3 preprocess/cmu_dog_preprocess.py ${data_path}/ITDD_data/src-test-tokenized.txt ${data_path}/ITDD_data/tgt-test-tokenized.txt ${data_path}/ITDD_data/knl-test-tokenized.txt
```



The `${data_path}` is location of raw dataset. So you could put the above three raw test datasets under `${data_path}` folder. The processed data will be  placed in the corresponding folder under `test_data` folder .

## \###RUN

### Train:

We train ZRKGC on 4 16GB Tesla V100 GPUs in a data parallel manner.

```
bash train.sh
```

Modify the `DATA_DIR`  in `train.sh` to what you download. The default path is `train_data`. Eval every 5000 step during training.  When current F1 is lower than previous F1, the training process will terminate.

This process will spend about 12hours with a batch size 10.

### Test:

```
 bash test.sh
```

Our model can be downloaded form from Google Drive (https://drive.google.com/drive/folders/178wJqriC-EQDnX7g3LGbwk0eWDI_CEcM?usp=sharing).  

Put it under the  `model` dir and modify the `MODEL_RECOVER_PATH`  in `test.sh`. 

To get automatic metrics on 3 test datasets (Wizard, Topical Chat and CMU_DoG),  you should install nlg-eval via (https://github.com/Maluuba/nlg-eval)  or download from Google Drive (https://drive.google.com/drive/folders/1Wejei90e-xPHNNABiPOEuG1JKjeXRwga?usp=sharing) .

This period will spend about 2 hours on a 2X8G 2080Ti machine with the default setting which produces the following results in the paper.

 

![Method PPL  ( WO W-seen)  ZRKGC 41.1  (WOW-unseen))  ZRKGC 42.7  (Topical Freq)  ZRKGC 45.1  (Topical Rare)  ZRKGC 42.8  (CMU_DoG)  ZRKGC 53.8  18.9  18.8  15.5  16.1  12.2  BLEU-I  0.225  0.220  0.218  0.223  0.161  BLEU-2  0.084  0.083  0.074  0.080  0.052  BLEU-3  0.039  0.040  0.033  0.037  0.021  BLEU-4  0.020  0.021  0.017  0.019  0.009  Average  0.891  0.890  0.894  0.887  0.838  Extrema  0.436  0.437  0.423  0.415  0.372  Greedy  0.683  0.680  0.678  0.672  0.639 ](file:////Users/llx/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/F0D5536F-78B5-B14A-890B-DE85A2D370A5.png)



Due to the time limitation, some details may be lost and we will refine this file later so that others could reproduce the results more easily.  At present, if there is anything unclear (e.g. the arguments or implementations), please refer to the scripts for further details. Thank you so much.