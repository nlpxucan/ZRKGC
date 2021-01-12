import sys
import random
import re
from metrics import bleu_metric
import numpy as np
import nltk

from tokenization import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("unilm_v2_bert_pretrain", do_lower_case=True)

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def move_stop_words(str):
	item = " ".join([w for w in str.split() if not w.lower() in stop_words])
	return item

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

def normalize_answer(s):
	"""Lower text and remove punctuation, articles and extra whitespace."""

	def remove_articles(text):
		return re_art.sub(' ', text)

	def white_space_fix(text):
		return ' '.join(text.split())

	def remove_punc(text):
		return re_punc.sub(' ', text)  # convert punctuation to spaces

	def lower(text):
		return text.lower()

	return white_space_fix(remove_articles(remove_punc(lower(s))))

def truncate(str, num):
	str = str.strip()
	length = len(str.split())
	list = str.split()[max(0, length - num):]
	return " ".join(list)

def detokenize(tk_str):
	tk_list = tk_str.strip().split()
	r_list = []
	for tk in tk_list:
		if tk.startswith('##') and len(r_list) > 0:
			r_list[-1] = r_list[-1] + tk[2:]
		else:
			r_list.append(tk)
	return " ".join(r_list)

src_path = sys.argv[1]
tgt_path = sys.argv[2]
knl_path = sys.argv[3]

with open(src_path, encoding="utf-8") as file:
	SRC = file.readlines()
with open(tgt_path, encoding="utf-8") as file:
	TGT = file.readlines()
with open(knl_path, encoding="utf-8") as file:
	KNL = file.readlines()

def main():
	mean_len_knows = 0
	with open("test_data/cmu_dog/test_cmu_dog.src.tk", "w", encoding="utf-8") as out1, open("test_data/cmu_dog/test_cmu_dog.tgt.tk", "w", encoding="utf-8") as out2:
		for i in range(len(SRC)):
			query_line = SRC[i].strip().replace(" &lt; SEP &gt; ", "<#Q#>").replace("&apos;", "'")
			tgt_line = TGT[i].strip().replace("&apos;", "'")
			# choice no.3
			knows = nltk.sent_tokenize(
				KNL[i].strip().split(" &lt; SEP &gt; ")[2].replace("&apos;", "'")) + nltk.sent_tokenize(
				KNL[i].strip().split(" &lt; SEP &gt; ")[0].replace("&apos;", "'")) + nltk.sent_tokenize(KNL[i].strip().split(" &lt; SEP &gt; ")[1].replace("&apos;", "'"))

			max_b2 = 0
			check_sentence = ""

			for know_line in knows:
				pro_know = normalize_answer(move_stop_words(know_line.strip()))
				pro_response = normalize_answer(move_stop_words(tgt_line.strip()))
				b1, b2, b3 = bleu_metric([pro_know], [pro_response])
				if b2 >= max_b2:
					max_b2 = b2
					check_sentence = know_line

			mean_len_knows += len(knows)
			use_know_list = knows
			if check_sentence in use_know_list:
				index = use_know_list.index(check_sentence)
				use_know_list[0], use_know_list[index] = use_know_list[index], use_know_list[0]
			else:
				use_know_list[0] = check_sentence
			assert use_know_list.index(check_sentence) == 0

			used_know_line = " <#K#> ".join(use_know_list)

			src_line = query_line + " <#Q2K#> " + used_know_line

			out1.write(" ".join(tokenizer.tokenize(src_line.strip())) + "\n")
			out2.write(" ".join(tokenizer.tokenize(tgt_line.strip())) + "\n")

			if i % 1000 == 0:
				print("have process {} data / {}".format(i, len(SRC)))


def process_ks(src_path, out_path):
	with open(src_path, "r", encoding="utf-8") as file:
		src = file.readlines()
	with open(out_path, "w", encoding="utf-8") as out:
		for i in range(len(src)):
			query = truncate(src[i].strip().split("<#Q2K#>")[0].strip(), 128)
			know_list = src[i].strip().split("<#Q2K#>")[1].split("<#K#>")

			for t in range(len(know_list)):
				line = query.strip()
				line += " <#Q2K#> "
				line += know_list[t].strip()

				out.write(line.strip())
				out.write("\n")

	print("done")


def detokenize_file(file_path):
	with open(file_path + ".tk", encoding="utf-8") as file:
		data = file.readlines()
	with open(file_path, "w", encoding="utf-8") as out:
		for i in range(len(data)):
			out.write(detokenize(data[i].strip()) + "\n")

if __name__ == "__main__":
	main()

	data_type = "cmu_dog"
	src_path = "test_data/{}/test_cmu_dog.src.tk".format(data_type)
	out_path = "test_data/{}/test_cmu_dog.ks.tk".format(data_type)
	process_ks(src_path,out_path)

	detokenize_file("test_data/{}/test_cmu_dog.tgt".format(data_type))
