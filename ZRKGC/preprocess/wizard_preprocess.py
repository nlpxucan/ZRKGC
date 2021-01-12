import os
import sys
import json
from wizard_generator import data_generator
from tokenization import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("unilm_v2_bert_pretrain", do_lower_case=True)

data_path = sys.argv[1]

text_truncate=128
max_knowledge=10000
knowledge_truncate=32
label_truncate=32
max_query_turn=4

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


if "random" in data_path:
	data_type="wizard_random"
elif "topic" in data_path:
	data_type = "wizard_topic"
else:
	print("WRONG DATATYPE")

def process_wizard(data_file):

	HISTORY = []
	KNOWLEDGE = []
	LABEL = []

	for (history, label, knowledge_sentences) in data_generator(data_file):
		# process context
		history_words = history.split()
		if len(history_words) > text_truncate:
			history_words = history_words[-text_truncate:]
		HISTORY.append(" ".join(history_words).strip())

		# process knowledge
		knowledge_list=[]
		if len(knowledge_sentences) > max_knowledge:
			keepers = 1 + np.random.choice(len(knowledge_sentences) - 1, max_knowledge, False)
			keepers[0] = 0
			knowledge_sentences = [knowledge_sentences[i] for i in keepers]

		for k in knowledge_sentences:
			fw_words = k.split()
			if len(fw_words) > knowledge_truncate:
				fw_words = fw_words[:knowledge_truncate]
			sentence = " ".join(fw_words).strip().split("__knowledge__")[1].strip()
			knowledge_list.append(sentence)
		KNOWLEDGE.append(" <#K#> ".join(knowledge_list))

		# process response
		label_words = label.split()
		if len(label_words) > label_truncate:
			label_words = label_words[:label_truncate]
		LABEL.append(" ".join(label_words).strip())

	assert len(HISTORY) == len(KNOWLEDGE) == len(LABEL)

	with open("test_data/{}/test_{}.src.tk".format(data_type, data_type), "w",encoding="utf-8") as out1, open("test_data/{}/test_{}.tgt.tk".format(data_type, data_type), "w", encoding="utf-8") as out2:
		for i in range(len(HISTORY)):

			if len(HISTORY[i].strip().split(" <#Q#> ")) > max_query_turn:
				history_line = " <#Q#> ".join(HISTORY[i].strip().split(" <#Q#> ")[-max_query_turn:])
			else:
				history_line = HISTORY[i].strip()
			assert len(history_line.strip().split(" <#Q#> ")) <= max_query_turn

			src_line = truncate(" ".join(tokenizer.tokenize(history_line)).strip(), text_truncate) + " <#Q2K#> " + " ".join(tokenizer.tokenize(KNOWLEDGE[i].strip())).strip()
			tgt_line = " ".join(tokenizer.tokenize(LABEL[i].strip())).strip()

			out1.write(src_line + "\n")
			out2.write(tgt_line + "\n")



def process_ks(src_path,out_path):
	with open(src_path,"r", encoding="utf-8") as file:
		src = file.readlines()
	with open(out_path,"w", encoding="utf-8") as out:
		for i in range(len(src)):
			query = truncate(src[i].strip().split("<#Q2K#>")[0].strip(),128)
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
	process_wizard(data_path)

	src_path = "test_data/{}/test_{}.src.tk".format(data_type, data_type)
	out_path = "test_data/{}/test_{}.ks.tk".format(data_type, data_type)
	process_ks(src_path,out_path)

	detokenize_file("test_data/{}/test_{}.tgt".format(data_type, data_type))
