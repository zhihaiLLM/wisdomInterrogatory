#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: search.py
@time: 2023/04/17
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""

import os
import json
from duckduckgo_search import ddg
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

class SourceService(object):
    def __init__(self, config):
        self.vector_store = None
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model_name)
        self.vector_store_path = self.config.vector_store_path
        self.kg_vector_stores = self.config.kg_vector_stores

    # def init_source_vector(self):
    #     """
    #     初始化本地知识库向量
    #     :return:
    #     """
    #     docs = []
    #     for i in list(self.kg_vector_stores.values()):
    #         docs = []
    #         for doc in os.listdir(i):
    #             if doc.endswith('.txt'):
    #                 print(doc)
    #                 loader = UnstructuredFileLoader(f'{i}/{doc}', mode="elements")
    #                 doc = loader.load()
    #                 docs.extend(doc)
    #         self.vector_store = FAISS.from_documents(docs, self.embeddings)
    #         self.vector_store.save_local(i)            

    def init_source_vector(self):
        """
        初始化本地知识库向量
        :return:
        """
        docs = []
        dic_all = {}
        for i in list(self.kg_vector_stores.values()):
            docs = []
            for doc in os.listdir(i):
                if doc.endswith('.json'):
                    print(doc)
                    with open(os.path.join(i,doc),"r") as f:
                        dic = json.load(f)
                        for key,val in dic.items():
                            docs.append(Document(page_content=key, metadata={"value": val}))
                            # dic_all[key] = val
                    # doc = loader.load()
                    # docs.extend(doc)
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            self.vector_store.save_local(i) 
        # return dic_all 

    def add_document(self, document_path):
        loader = UnstructuredFileLoader(document_path, mode="elements")
        doc = loader.load()
        self.vector_store.add_documents(doc)
        self.vector_store.save_local(self.vector_store_path)

    def load_vector_store(self, path):
        if path is None:
            self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings)
        else:
            self.vector_store = FAISS.load_local(path, self.embeddings)
        return self.vector_store

    def search_web(self, query):

        # SESSION.proxies = {
        #     "http": f"socks5h://localhost:7890",
        #     "https": f"socks5h://localhost:7890"
        # }
        try:
            results = ddg(query)
            web_content = ''
            if results:
                for result in results:
                    web_content += result['body']
            return web_content
        except Exception as e:
            print(f"网络检索异常:{query}")
            return ''
# if __name__ == '__main__':
#     config = LangChainCFG()
#     source_service = SourceService(config)
#     source_service.init_source_vector()
#     search_result = source_service.vector_store.similarity_search_with_score('科比')
#     print(search_result)
#
#     source_service.add_document('/home/searchgpt/yq/Knowledge-ChatGLM/docs/added/科比.txt')
#     search_result = source_service.vector_store.similarity_search_with_score('科比')
#     print(search_result)
#
#     vector_store=source_service.load_vector_store()
#     search_result = source_service.vector_store.similarity_search_with_score('科比')
#     print(search_result)
