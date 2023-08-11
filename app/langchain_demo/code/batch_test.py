# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import shutil
import random
import json
import torch
from app_modules.overwrites import postprocess
from app_modules.presets import *
from clc.langchain_application import LangChainApplication, torch_gc
from transformers import StoppingCriteriaList
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextIteratorStreamer, StoppingCriteriaList
from clc.callbacks import Iteratorize, Stream, _SentinelTokenStoppingCriteria, clear_torch_cache
from tqdm import tqdm
import torch.distributed as dist

device1 = torch.device("cuda:0")  # 使用第一张GPU卡
device2 = torch.device("cuda:1")  # 使用第二张GPU卡
# dist.init_process_group(backend="nccl")
# local_rank = torch.distributed.get_rank()
# torch.cuda.set_device(local_rank)
# DEVICE = torch.device("cuda", local_rank)
random.seed(2023)
# 修改成自己的配置！！！
# 102-113:危害国家安全罪，114-139:危害公共安全罪，
# 140-231:破坏社会主义市场经济秩序罪，232-262:侵犯公民人身权利、民主权力罪，
# 263-276:侵犯财产罪，277-367:妨害社会管理秩序罪，
# 368-381:危害国防利益罪，382-396:贪污贿赂罪，397-419:渎职罪，420-451:军人违反职责罪

class LangChainCFG:
    llm_model_name = '/root/data1/luwen/luwen_baichuan/output/zju_model_0710_100k_wenshu30k'  # 本地模型文件 or huggingface远程仓库
    llm_model_name2 = '/root/data1/luwen/luwen_llama/mymodel/alpaca_plus_13b_sft298k_hf'
    embedding_model_name = '/root/data1/liang/simcse_training/saved_model/new_embedding_random_large_4/11000'
    # embedding_model_name = '/root/data1/luwen/luwen_llama/mymodel/text2vec_large'  # 检索模型文件 or huggingface远程仓库
    kb_desc_path = "/root/data1/luwen/app/langchain_multi_stage/cache/kb_desc.json"
    vector_store_path = '/root/data1/luwen/app/langchain_demo/data/cache'
    kg_vector_stores = {
        '法律法条': '/root/data1/luwen/app/langchain_demo/data/cache/legal_article',
        '法律书籍': '/root/data1/luwen/app/langchain_demo/data/cache/legal_book',
        '法律文书模版':'/root/data1/luwen/app/langchain_demo/data/cache/legal_template',
        '法律案例': '/root/data1/luwen/app/langchain_demo/data/cache/legal_case',
        '法律考试': '/root/data1/luwen/app/langchain_demo/data/cache/judicialExamination',
    }  # 可以替换成自己的知识库，如果没有需要设置为None
    # kg_vector_stores=None
    patterns = ['模型问答', '知识库问答']  #
    # n_gpus=1
    device = {"dev1":device1,"dev2":device2}



config = LangChainCFG()
application = LangChainApplication(config)

# application.source_service.init_source_vector()


def predict(input,
            # chatbot,
            large_language_model,
            embedding_model,
            kg_names,
            history=None,
            max_length=None,
            top_k = 1,
            recall_only=False,
            multi_stage_pattern=False,
            **kwargs):
    # print(large_language_model, embedding_model)
    if large_language_model=="zju-lm":
        application.llm_service.tokenizer = application.tokenizer2
        application.llm_service.model = application.model2
    application.llm_service.max_token = max_length
    # print(input)
    if history == None:
        history = []
    search_text = ''

    application.llm_service.history = history[-5:]
    max_memory = 4096 - max_length
    kb_based = True if len(kg_names) else False

    if len(history) != 0:
        if large_language_model=="zju-bc":
            input = "".join(["</s>Human:\n" + i[0] +"\n\n" + "</s>Assistant:\n" + i[1] + "\n\n"for i in application.llm_service.history]) + \
            "</s>Human:\n" + input
            input = input[len("</s>Human:\n"):]
        else:
            input = "".join(["### Instruction:\n" + i[0] +"\n\n" + "### Response: " + i[1] + "\n\n" for i in application.llm_service.history]) + \
            "### Instruction:\n" + input
            input = input[len("### Instruction:\n"):]
    if len(input) > max_memory:
        input = input[-max_memory:]

    if kb_based:
        if multi_stage_pattern:
            kb_descs = application.past_vector_store.similarity_search_with_score(input, k=1) 
            kb_names = [application.reverse_dic[doc[0].page_content] for doc in kb_descs]    
            related_docs_with_score_seq = []
            for kg_name in kg_names:
                tmp_lst =[]
                for kb_name in kb_names:
                    if kg_name+kb_name+".txt" in application.vector_store_dic.keys():
                        curr_vector_store = application.vector_store_dic[kg_name+kb_name+".txt"]
                        tmp_lst+=curr_vector_store.similarity_search_with_score(input, k=top_k)
                related_docs_with_score_seq.append(tmp_lst)
        else:
            related_docs_with_score_seq = []
            for kg_name in kg_names:
                application.source_service.load_vector_store(application.config.kg_vector_stores[kg_name])
                related_docs_with_score_seq.append(application.source_service.vector_store.similarity_search_with_score(input, k=top_k))
        
        related_docs_with_score = related_docs_with_score_seq
        # if len(related_docs_with_score) > 0:
        #     input, context_with_score = application.generate_prompt(related_docs_with_score, input, large_language_model,kg_names)
        # search_text += context_with_score

        if recall_only:
            context_all = {}
            for i,kg_name in enumerate(kg_names):
                context_one = []
                for j,doc in enumerate(related_docs_with_score[i]):
                    context_one.append(doc[0].page_content)
                context_all[kg_name] = context_one
            return context_all

        if len(related_docs_with_score) > 0:
            input, context_with_score = application.generate_prompt(related_docs_with_score, input, large_language_model,kg_names)
        search_text += context_with_score
    torch_gc()
    result = application.llm_service._call(input, kb_based, model_name=large_language_model)

    return result

def get_prompt(input, prompt_lst):
    curr_prompt = random.choice(prompt_lst)
    input = curr_prompt + "\n"+ input
    return input


embedding_model = "text2vec"

large_language_model = "zju-bc"

max_length = 512

top_k = 1

multi_stage_pattern = False

kg_names = ["法律法条","法律案例"]
# kg_names = ["刑法法条"]

prompt_dic = {"article":["请问下面案件中被告人触犯的具体法律是哪一条？"],
              "charge":["请问下面案件中被告人犯的是什么罪？"],
              "penalty":["请预测下面案件中被告人可能的刑期是多少个月？"]}
# prompt_dic={"article":["请列举案件中被告人触犯的具体法律是哪一条。","被告人在刑法中触犯了哪些法条？","能否告诉我被告人在《中华人民共和国刑法》中触犯了哪一条法律？","请提供被告人触犯的具体法律条款。","在案例中，涉及了哪些刑法法条？","能告诉我被告人犯了《中华人民共和国刑法》里的哪一条吗？"],
# "charge":["案例中涉及了哪些具体刑法罪名？","请提供被告人的刑法罪名。","写出案件中被告人犯的是什么罪","能告诉我被告人犯了《中华人民共和国刑法》里的什么罪吗？","案例中触犯的刑法罪名有哪些？"],
# "penalty":["请预测一下案件中被告人可能的刑期。","被告人会面临多久的有期徒刑？","案件中被告人可能会获得什么刑罚？","请分析案件中被告人可能面对的刑期。","被告人可能会被判多长刑期？"]}
test_path = "/root/data1/luwen/app/langchain_demo/data/test_sample.json"
results=[]
with open(test_path,"r") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        data = json.loads(line)
        fact_origin = data["input"]
        fact = get_prompt(fact_origin, prompt_dic["charge"])
        result = predict(input=fact,large_language_model=large_language_model,embedding_model=embedding_model,
                        kg_names=kg_names,max_length=max_length,top_k=top_k,multi_stage_pattern=multi_stage_pattern)
        results.append({"fact":fact_origin,"label":data["label"],"pred":result})
with open("/root/data1/luwen/app/langchain_demo/data/result/generation.txt", "w") as f:
    for dic in results:
        json.dump(dic,f,ensure_ascii=False)
        f.write("\n")