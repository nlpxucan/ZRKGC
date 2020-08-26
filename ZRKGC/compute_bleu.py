import sys, os
import random
import json
import nltk
import random
import requests
from metrics import  bleu_metric, knowledge_metric,f_one,normalize_answer,bleu

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def move_stop_words(str):

    item = " ".join([w for w in str.split() if not w.lower() in stop_words])
    return item


def detokenize(tk_str):
	tk_list = tk_str.strip().split()
	r_list = []
	for tk in tk_list:
		if tk.startswith('##') and len(r_list) > 0:
			r_list[-1] = r_list[-1] + tk[2:]
		else:
			r_list.append(tk)
	return " ".join(r_list)


def compute_bleu_between_rule_and_model(rule_parh,model_path):
    with open(rule_parh, "r", encoding='utf-8') as rule_file:
        rule = rule_file.readlines()
    with open(model_path, "r", encoding='utf-8') as model_file:
        model = model_file.readlines()
    model = [detokenize(item.lower()) for item in model]


    b1, b2, b3 = bleu_metric(rule,model)

    print("b1:{},b2:{},b3:{}".format(round(b1,4),round(b2,4),round(b3,4)))

    res = f_one(rule,model)
    print('f1:{}'.format(res[0]))


if __name__ == "__main__":

	check_path = sys.argv[1]
	dataset = sys.argv[2]
	data_path = "test_data/{}/test_{}.tgt.tk".format(dataset, dataset)
	compute_bleu_between_rule_and_model(check_path,data_path)






