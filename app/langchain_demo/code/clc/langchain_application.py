import torch
from clc.gpt_service import ChatGLMService
from clc.source_service import SourceService
from typing import List
from clc.matching import init_all_articles

def torch_gc():
    if torch.cuda.is_available():
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
        self.llm_service.load_model(model_name_or_path=self.config.llm_model_name)
        self.source_service = SourceService(config)
        self.all_articles, self.choices = init_all_articles()

        print("trying to load source vector store ")
        try:
            self.source_service.load_vector_store(self.config.vector_store_path)
        except Exception as e:
            self.source_service.init_source_vector()

    def generate_prompt(self, related_docs,
                        query: str, model_name: str, kg_names: List[str]) -> str:
        if model_name=="zju-bc":
            prompt_template = ("</s>Human:\n{question}\n{context}")
        else:
            prompt_template = ("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{question}\n{context}")
        context_all = ""
        for i,kg_name in enumerate(kg_names):
            context_one = []
            for j,doc in enumerate(related_docs[i]):
                context_one += ["[{}] ".format(j+1) + doc[0].metadata["value"]]
            context = "\n".join(context_one)
            if context:
                context = "可供参考的"+kg_name + ":" + context + "\n"
                context_all += context
        context_with_score = context_all
        prompt = prompt_template.replace("{question}", query).replace("{context}", context_all)
        return prompt, context_with_score
