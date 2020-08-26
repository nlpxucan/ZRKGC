import sys
import json
import re
import nltk
import random
from nltk.tokenize import word_tokenize
from metrics import bleu_metric, normalize_answer
from nltk.corpus import stopwords
from tokenization import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("unilm_v2_bert_pretrain", do_lower_case=True)


stop_words = set(stopwords.words('english'))

def move_stop_words(str):
	item = " ".join([w for w in str.split() if not w.lower() in stop_words])
	return item


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


dialog_path = sys.argv[1]  #"../conversations/{}.json".format(part)
know_path = sys.argv[2] #"../reading_sets/post-build/{}.json".format(part)

if "rare" in dialog_path:
	part = "topical_rare"
else:
	part = "topical_freq"

# generate conversation and knowledge
with open(dialog_path, 'r', encoding='utf-8') as f:
	conversations_file = json.load(f)
with open(know_path, 'r', encoding='utf-8') as f:
	know_file = json.load(f)

def main():
	all_query = []
	all_know = []
	all_other_know = []

	count = 0
	no_know = 0
	for dialog_id, dialog_turn in conversations_file.items():
		count += 1
		dialog_content = dialog_turn["content"]
		knowledge_turn = know_file[dialog_id]

		query_line = ""
		for every_content in dialog_content:
			sentence = every_content["message"]
			sentence = sentence.encode('unicode_escape').decode('utf-8')

			know_ids = every_content["knowledge_source"]  # list

			single_knowledges = knowledge_turn[every_content["agent"]]
			article_knowledge = knowledge_turn["article"]
			other_know_ids = ["FS1", "FS2", "FS3", "AS1", "AS2", "AS3"]
			other_know_ids = list(set(other_know_ids) - set(know_ids))
			knowledge_list = []
			other_knowledge_list = []
			for know_id in know_ids:
				if know_id == "Personal Knowledge":
					pass
				elif know_id in single_knowledges:
					assert "shortened_wiki_lead_section" or "summarized_wiki_lead_section" in single_knowledges[
						know_id]
					if "shortened_wiki_lead_section" in single_knowledges[know_id]:
						knowledge_list.extend(
							nltk.sent_tokenize(single_knowledges[know_id]["shortened_wiki_lead_section"]))
					else:
						knowledge_list.extend(
							nltk.sent_tokenize(single_knowledges[know_id]["summarized_wiki_lead_section"]))
					for item in single_knowledges[know_id]["fun_facts"]:
						knowledge_list.append(item)
				elif know_id in article_knowledge:
					knowledge_list.extend(nltk.sent_tokenize(article_knowledge[know_id]))

			for other_know_id in other_know_ids:
				if other_know_id in single_knowledges:
					assert "shortened_wiki_lead_section" or "summarized_wiki_lead_section" in single_knowledges[know_id]
					if "shortened_wiki_lead_section" in single_knowledges[other_know_id]:
						other_knowledge_list.extend(
							nltk.sent_tokenize(single_knowledges[other_know_id]["shortened_wiki_lead_section"]))
					else:
						other_knowledge_list.extend(
							nltk.sent_tokenize(single_knowledges[other_know_id]["summarized_wiki_lead_section"]))
					for item in single_knowledges[other_know_id]["fun_facts"]:
						other_knowledge_list.append(item)

				elif (dialog_turn["config"] != "C") and other_know_id in article_knowledge:
					other_knowledge_list.extend(nltk.sent_tokenize(article_knowledge[other_know_id]))

			if knowledge_list == []:
				no_know += 1
				knowledge_list = ["__no_knowledge__"]

			know_line = ""
			for k in knowledge_list:
				k = k.encode('unicode_escape').decode('utf-8')
				know_line += k
				know_line += "\t"
			all_know.append(know_line)

			other_know_line = ""
			for k in other_knowledge_list:
				k = k.encode('unicode_escape').decode('utf-8')
				other_know_line += k
				other_know_line += "\t"
			all_other_know.append(other_know_line)

			query_line += sentence
			query_line += " <#Q#> "
		all_query.append(query_line)

	assert len(all_other_know) == len(all_know)

	num = 0
	src = []
	tgt = []
	for i in range(len(all_query)):

		query_list = all_query[i].strip().split("<#Q#>")[:-1]
		for t in range(len(query_list)):
			history = " <#Q#> ".join(query_list[:t])
			if history.strip() == "":
				history = "__no_history__"

			knows = all_know[num].strip().split("\t")

			max_b2 = 0
			for one_know in knows:
				b1, b2, b3 = bleu_metric([normalize_answer(move_stop_words(query_list[t].strip()))], [normalize_answer(move_stop_words(one_know))])
				if b2 >= max_b2:
					max_b2 = b2
					check = one_know
			assert check in knows

			loc = knows.index(check)
			knows[loc], knows[0] = knows[0], knows[loc]

			other_knows = all_other_know[num].strip().split("\t")[:-1]

			know_str = " <#K#> ".join(knows)
			src_line = history.strip() + " <#Q2K#> " + know_str.strip()
			tgt_line = query_list[t].strip()

			src.append(src_line)
			tgt.append(tgt_line)
			num += 1


	assert num == len(all_know)


	with open("test_data/{}/test_{}.src.tk".format(part, part), "w") as src_out, \
			open("test_data/{}/test_{}.tgt.tk".format(part, part), "w") as tgt_out:

		mean_know = 0
		for i in range(len(src)):

			query_list = src[i].strip().split("<#Q2K#>")[0].split("<#Q#>")
			query_list = [" ".join(word_tokenize(item.strip())) for item in query_list]
			query_line = " <#Q#> ".join(query_list).strip()

			know_list = src[i].strip().split("<#Q2K#>")[1].split("<#K#>")
			know_list = [" ".join(word_tokenize(item.strip())) for item in know_list]
			mean_know += len(know_list)
			know_line = " <#K#> ".join(know_list).strip()

			pro_src_line = " ".join(tokenizer.tokenize(query_line + " <#Q2K#> " + know_line))
			pro_tgt_line = " ".join(tokenizer.tokenize(" ".join(word_tokenize(tgt[i].strip()))))

			src_out.write(pro_src_line)
			src_out.write("\n")

			tgt_out.write(pro_tgt_line)
			tgt_out.write("\n")

			if i % 1000 == 0:
				print("have process {} data".format(i))


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

	src_path = "test_data/{}/test_{}.src.tk".format(part, part)
	out_path = "test_data/{}/test_{}.ks.tk".format(part, part)
	process_ks(src_path,out_path)

	detokenize_file("test_data/{}/test_{}.tgt".format(part, part))