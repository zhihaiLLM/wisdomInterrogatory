import os
import json
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

    def init_source_vector(self):
        """
        初始化本地知识库向量
        :return:
        """
        docs = []
        for i in list(self.kg_vector_stores.values()):
            docs = []
            for doc in os.listdir(i):
                if doc.endswith('.json'):
                    print(doc)
                    with open(os.path.join(i,doc),"r") as f:
                        lines = f.readlines()
                        for line in lines:
                            data = json.loads(line)
                            key = data["key"]
                            val = data["value"]
                            docs.append(Document(page_content=key, metadata={"value": val}))
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            self.vector_store.save_local(i) 

    def load_vector_store(self, path):
        if path is None:
            self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings)
        else:
            self.vector_store = FAISS.load_local(path, self.embeddings)
        return self.vector_store