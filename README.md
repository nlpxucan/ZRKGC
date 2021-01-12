# Zero-Resource Knowledge Grounded Dialogue Generation

 

## Dependency

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
## Data


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

### Train data

Download and unzip from Google Drive (https://drive.google.com/file/d/1bKjHtJMDwxsXRwQD2UQg37TdEfVg6l3k/view?usp=sharing).  Because of the large size of train data , this process will take some time.


## RUN

### Test:

```
 bash test.sh
```

Our model can be downloaded form from Google Drive (https://drive.google.com/drive/folders/178wJqriC-EQDnX7g3LGbwk0eWDI_CEcM?usp=sharing).  

Put it under the  `model` dir and modify the `MODEL_RECOVER_PATH`  in `test.sh`. 

To get automatic metrics on 3 test datasets (Wizard, Topical Chat and CMU_DoG),  you should install nlg-eval via (https://github.com/Maluuba/nlg-eval)  or download from Google Drive (https://drive.google.com/drive/folders/1Wejei90e-xPHNNABiPOEuG1JKjeXRwga?usp=sharing) .

This period will spend about 2 hours on a 2X8G 2080Ti machine with the default setting which produces the following results in the paper.

### Train:

Download from (https://github.com/microsoft/unilm/tree/master/s2s-ft) to get `[unilm1.2-base-uncased]` model. The model are trained by using the same model configuration and WordPiece vocabulary as BERT Base.

Note that  `unilm_v2_bert_pretrain` folder shoud  contains  three components: `bert_config.json` 、`unilm1.2-base-uncased.bin`  and  `vocab.txt`. And pls replace 'vocab.txt' with this from Google Drive (https://drive.google.com/file/d/1q-4QBE_H0fulb7_izEABOLbfu1jPGpJZ/view?usp=sharing).


We train ZRKGC on 4 16GB Tesla V100 GPUs in a data parallel manner.

```
bash train.sh
```

Modify the `DATA_DIR`  in `train.sh` to what you download. The default path is `train_data`. Eval every 5000 step during training.  When current F1 is lower than previous F1, the training process will terminate.

This process will spend about 12hours with a batch size 10.

## Citation

If you find ZRKGC useful in your work, you can cite the following paper:
```
@article{li2020zero,
  title={Zero-Resource Knowledge-Grounded Dialogue Generation},
  author={Li, Linxiao and Xu, Can and Wu, Wei and Zhao, Yufan and Zhao, Xueliang and Tao, Chongyang},
  journal={arXiv preprint arXiv:2008.12918},
  year={2020}
}
```
