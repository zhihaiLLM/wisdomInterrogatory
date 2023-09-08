import torch
from clc.gpt_service import BaichuanService
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
        self.llm_service = BaichuanService()
        self.llm_service.load_model(model_name_or_path=self.config.llm_model_name)
        self.source_service = SourceService(config)
        self.all_articles, self.choices = init_all_articles()

        print("trying to load source vector store ")
        try:
            self.source_service.load_vector_store(self.config.vector_store_path)
        except Exception as e:
            self.source_service.init_source_vector()

    def generate_prompt(self, related_docs,
                        query: str, kg_names: List[str]) -> str:
        prompt_template = ("{question}\n{context}")
        context_all, context_all_inner = "", ""
        for i,kg_name in enumerate(kg_names):
            context_one, context_one_inner = [], []
            for j,doc in enumerate(related_docs[i]):
                if doc[1] < 500:
                    context_one += ["[{}] ".format(j+1) + doc[0].metadata["value"]]
                    context_one_inner += [doc[0].metadata["value"]]
            if len(context_one)>0:
                context_all = context_all + "可供参考的"+kg_name + ":\"" + "\n".join(context_one) + "\"\n"
                context_all_inner = context_all_inner + "可供参考的" + kg_name + ":\"" + "\n".join(context_one_inner) + "\"\n"
            else:
                context_all = context_all + kg_name+ "中未找到相关知识"+ "\n"
        if len(context_all_inner)>0:
            context_all_inner = "下面提供可能与对话相关的资料，你可以参考以下资料如果它们与上面的文本内容有关。\n" + context_all_inner
            prompt = prompt_template.replace("{question}", query).replace("{context}", context_all_inner)
        else:
            prompt = query
        return prompt, context_all
