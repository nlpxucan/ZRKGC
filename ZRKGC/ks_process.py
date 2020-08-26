import sys
import heapq
import random
import operator

dataset = sys.argv[1]

def truncate(str, num):
    str = str.strip()
    length = len(str.split())
    list = str.split()[max(0, length - num):]
    return " ".join(list)


def compute_selection_score(data_path, src_path, top):
    with open(src_path, "r", encoding="utf-8") as file:
        src = file.readlines()

    with open(data_path, "r", encoding="utf-8") as file:
        data = file.readlines()

    query_know_dict = {}
    t = 0
    for i in range(len(src)):

        query = src[i].strip().split("<#Q2K#>")[0]
        know_list = src[i].strip().split("<#Q2K#>")[1].split("<#K#>")
        for num in range(len(know_list)):
            try:
                query_know_dict[str(i) + "\t" + query].append(data[t].strip().split("<#Q2K#>")[1].strip())
                t += 1
            except:
                query_know_dict[str(i) + "\t" + query] = []
                query_know_dict[str(i) + "\t" + query].append(data[t].strip().split("<#Q2K#>")[1].strip())
                t += 1

    assert t == len(data)

    count = 0
    for query, knows in query_know_dict.items():
        check_sent = knows[0].split("\t")[0]
        random.shuffle(knows)

        bleu_list = []
        know_list = []
        for know in knows:
            try:
                bleu_list.append(float(know.split("\t")[1]))
                know_list.append(know.split("\t")[0])
            except:
                pass

        max_num_index_list = map(bleu_list.index, heapq.nlargest(top, bleu_list))
        if know_list[list(max_num_index_list)[0]] == check_sent:
            count += 1

        max_num_index_list = map(bleu_list.index, heapq.nlargest(top, bleu_list))
    print(count / len(query_know_dict))


def get_rank_know(data_path, src_path, out_path):
    with open(data_path, "r", encoding="utf-8") as file:
        data = file.readlines()

    with open(src_path, "r", encoding="utf-8") as file:
        src = file.readlines()

    query_know_dict = {}
    t = 0
    for i in range(len(src)):

        query = src[i].strip().split("<#Q2K#>")[0]
        know_list = src[i].strip().split("<#Q2K#>")[1].split("<#K#>")
        for num in range(len(know_list)):
            try:
                query_know_dict[str(i) + "\t" + query].append(data[t].strip().split("<#Q2K#>")[1].strip())
                t += 1
            except:
                query_know_dict[str(i) + "\t" + query] = []
                query_know_dict[str(i) + "\t" + query].append(data[t].strip().split("<#Q2K#>")[1].strip())
                t += 1

    assert t == len(data)

    with open(out_path, "w", encoding="utf-8") as out:
        count = 0
        for query, knows in query_know_dict.items():
            check_sent = knows[0].split("\t")[0]

            random.shuffle(knows)

            bleu_list = []
            know_list = []
            for know in knows:
                try:
                    bleu_list.append(float(know.split("\t")[1]))
                    know_list.append(know.split("\t")[0])
                except:
                    pass

            know_bleu_map = {}
            for i in range(len(bleu_list)):
                know_bleu_map[know_list[i]] = bleu_list[i]
            sorted_knows = sorted(know_bleu_map.items(), key=operator.itemgetter(1), reverse=True)

            line = truncate(query.strip().split("\t")[1].strip(), 128)
            line += " <#Q2K#> "

            for t in range(len(sorted_knows)):
                line += sorted_knows[t][0]
                line += " <#K#> "

            line = " ".join(line.strip().split()[:-1])
            line = line.replace(" 。", "")
            line = " ".join(line.strip().split()[:210])

            if check_sent in line or line.split("<#K#>")[-1] in check_sent:
                count += 1

            out.write(line)
            out.write("\n")
        print(len(query_know_dict))
        print(count / len(query_know_dict))


print(dataset)
src_path = "test_data/{}/test_{}.src.tk".format(dataset, dataset)
data_path = "test_data/{}/test_{}.ks_score.tk".format(dataset, dataset)
compute_selection_score(data_path, src_path, 1)

data_path = "test_data/{}/test_{}.ks_score.tk".format(dataset, dataset)
out_path = "test_data/{}/rank_test_{}.src.tk".format(dataset, dataset)

get_rank_know(data_path, src_path, out_path)
