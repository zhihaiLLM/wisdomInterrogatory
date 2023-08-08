#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: model.py
@time: 2023/04/17
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
import torch
from clc.config import LangChainCFG
from clc.gpt_service import ChatGLMService
from clc.source_service import SourceService
from typing import List

def torch_gc():
    if torch.cuda.is_available():
        # with torch.cuda.device(DEVICE):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        try:
            from torch.mps import empty_cache
            empty_cache()
        except Exception as e:
            print(e)
            print("如果您使用的是 macOS 建议将 pytorch 版本升级至 2.0.0 或更高版本，以支持及时清理 torch 产生的内存占用。")


class LangChainApplication(object):
    def __init__(self, config):
        self.config = config
        self.llm_service = ChatGLMService()
        self.llm_service.load_model(model_name_or_path=self.config.llm_model_name, device=self.config.device["dev1"])
        # self.tokenizer2 ,self.model2 = self.llm_service.load_model2(model_name_or_path=self.config.llm_model_name2, device=self.config.device["dev2"])
        # self.llm_service.load_model_on_gpus(model_name_or_path=self.config.llm_model_name,num_gpus=self.config.n_gpus)
        self.source_service = SourceService(config)

        # if self.config.kg_vector_stores is None:
        #     print("init a source vector store")
        #     self.source_service.init_source_vector()
        # else:
        print("trying to load source vector store ")
        try:
            self.source_service.load_vector_store(self.config.vector_store_path)
        except Exception as e:
            self.dic_all = self.source_service.init_source_vector()

    def generate_prompt(self, related_docs,
                        query: str, model_name: str, kg_names: List[str]) -> str:
        if model_name=="zju-bc":
            # prompt_template = ("</s>Related knowledge:\n{context}\n\n</s>Human:\n{question}")
            prompt_template = ("</s>Human:\n{question}\n{context}")
        else:
            # prompt_template = ("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            # "### Related knowledge:\n{context}\n\n### Instruction:\n{question}")
            prompt_template = ("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{question}\n{context}")
        context_all = ""
        for i,kg_name in enumerate(kg_names):
            context_one = []
            for j,doc in enumerate(related_docs[i]):
                # context_one += ["[{}] ".format(j+1) + doc[0].page_content]
                # 用知识的key检索，拼上value
                context_one += ["[{}] ".format(j+1) + doc[0].metadata["value"]]
            context = "\n".join(context_one)
            if context:
                context = "可供参考的"+kg_name + ":" + context + "\n"
                context_all += context
        context_with_score = context_all
        # context_with_score = "\n".join(["[{}] ".format(i+1)+doc[0].page_content for i,doc in enumerate(related_docs)])
        # context_with_score = "\n".join(["[{}] ".format(i)+"content:"+doc[0].page_content+ " scores:"+str(doc[1]) for i,doc in enumerate(related_docs)])
        prompt = prompt_template.replace("{question}", query).replace("{context}", context_all)
        return prompt, context_with_score

    def get_knowledge_based_answer(self, query,
                                   history_len=5,
                                   temperature=0.1,
                                   top_p=0.9,
                                   top_k=3,
                                   web_content='',
                                   chat_history=[],
                                   kg_names = [],
                                   model_name = " "):
        # if web_content:
        #     prompt_template = f"""基于以下已知信息，简洁和专业的来回答用户的问题。
        #                         如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
        #                         已知网络检索内容：{web_content}""" + """
        #                         已知内容:
        #                         {context}
        #                         问题:
        #                         {question}"""
        # else:
            # prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
            #                                 如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
            #                                 已知内容:
            #                                 {context}
            #                                 问题:
            #                                 {question}"""
        # prompt = PromptTemplate(template=prompt_template,
        #                         input_variables=["context", "question"])
        self.llm_service.history = chat_history[-history_len:] if history_len > 0 else []

        self.llm_service.temperature = temperature
        self.llm_service.top_p = top_p

        if len(kg_names) == 0:
            kg_names = ["刑法法条"]
        related_docs_with_score_seq = []
        for kg_name in kg_names:
            self.source_service.load_vector_store(self.config.kg_vector_stores[kg_name])
            related_docs_with_score_seq.append(self.source_service.vector_store.similarity_search_with_score(query, k=top_k))
        related_docs_with_score = related_docs_with_score_seq

        torch_gc()

        if len(self.llm_service.history) != 0:
            if model_name=="zju-bc":
                input = "".join(["</s>Human:\n" + i[0] +"\n\n" + "</s>Assistant:\n" + i[1] + "\n\n"for i in self.history]) + \
                "</s>Human:\n" + input
                input = input[len("</s>Human:\n"):]
            else:
                input = "".join(["### Instruction:\n" + i[0] +"\n\n" + "### Response: " + i[1] + "\n\n" for i in self.llm_service.history]) + \
                "### Instruction:\n" + input
                input = input[len("### Instruction:\n"):]
        else:
            input = query 

        if len(related_docs_with_score) > 0:
            prompt, context_with_score = self.generate_prompt(related_docs_with_score, input, model_name)
        else:
            prompt = input

        result = self.llm_service._call(prompt, kb_based=True, model_name=model_name, )

        # knowledge_chain = RetrievalQA.from_llm(
        #     llm=self.llm_service,
        #     retriever=self.source_service.vector_store.as_retriever(
        #         search_kwargs={"k": top_k}),
        #     prompt=prompt)
        # knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
        #     input_variables=["page_content"], template="{page_content}")

        # knowledge_chain.return_source_documents = True

        # result = knowledge_chain({"query": query})
        return result, context_with_score

    def get_llm_answer(self, query='', web_content='',chat_history=[], model_name="", history_len=5):
        self.llm_service.history = chat_history[-history_len:] if history_len > 0 else []
        if web_content:
            prompt = f'基于网络检索内容：{web_content}，回答以下问题{query}'
        else:
            prompt = query
        # print("history in app:", history)
        result = self.llm_service._call(input=prompt, model_name=model_name)
        return result


if __name__ == '__main__':
    config = LangChainCFG()
    application = LangChainApplication(config)
    # result = application.get_knowledge_based_answer('马保国是谁')
    # print(result)
    # application.source_service.add_document('/home/searchgpt/yq/Knowledge-ChatGLM/docs/added/马保国.txt')
    # result = application.get_knowledge_based_answer('马保国是谁')
    # print(result)
    result = application.get_llm_answer('马保国是谁')
    print(result)
