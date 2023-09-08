import json
from fuzzywuzzy import process
import os
import re

def key_words_match_intention(input):
    kg_names = set()
    with open("app/langchain_demo/data/cache/intention_reg.json", "r") as f:
        dic = json.load(f)
        for key,val in dic.items():
            for el in val:
                if el in input:
                    kg_names.add(key)
    return kg_names

def init_all_articles():
    art_dir = "app/langchain_demo/data/cache/legal_articles"
    dic_all = {}
    choices = []
    for doc in os.listdir(art_dir):
        if doc.endswith('.json'):
            # print(doc)
            with open(os.path.join(art_dir,doc),"r") as f:
                lines = f.readlines()
                for line in lines:
                    data = json.loads(line)
                    dic_all[data["key0"]] = data["value"]
                    choices.append(data["key0"].split(" ")[0]) 

    choices = list(set(choices))
    return dic_all, choices

def key_words_match_knowledge(dic_all, choices, query):
    result_title = process.extract(query, choices, limit=1)
    match = re.search(r'第([一二三四五六七八九零百十]+)条', query)
    if match:
        result_num = match.group(1)
        key0 = result_title[0][0]+" 第"+ result_num +"条"
        if key0 in dic_all.keys():
            return (key0, dic_all[key0])
    return 