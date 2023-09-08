#!/usr/bin/env python
# -*- coding:utf-8 _*-
import os
from typing import Dict, Union, Optional
from typing import List
import torch
from accelerate import load_checkpoint_and_dispatch
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextIteratorStreamer, StoppingCriteriaList


class BaichuanService(LLM):
    max_token: int = 10000
    temperature: float = 0.2
    top_p = 0.9
    history = []
    tokenizer: object = None
    model: object = None
    max_memory = 1000

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "Baichuan"

    def _call(self,
              input: str,
              kb_based: bool = False,
              model_name: Optional[List[str]] = None,
              stop: Optional[List[str]] = None) -> str:

        self.max_memory = 4096 - int(self.max_token or 0)
        now_input = input
        if not kb_based:
            if len(self.history) != 0:

                if model_name=="zju-bc":
                    input = "".join(["</s>Human:\n" + i[0] +"\n\n" + "</s>Assistant:\n" + i[1] + "\n\n"for i in self.history]) + \
                    "</s>Human:\n" + input
                    input = input[len("</s>Human:\n"):]
                else:
                    input = "".join(["### Instruction:\n" + i[0] +"\n\n" + "### Response: " + i[1] + "\n\n" for i in self.llm_service.history]) + \
                    "### Instruction:\n" + input
                    input = input[len("### Instruction:\n"):]
                
        if len(input) > self.max_memory:
            input = input[-self.max_memory:]
        # print("input",input)
        print("histroy in call:", self.history)
        prompt = self.generate_prompt(input, kb_based, model_name)
        print("prompt:  ",prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        # input_ids = inputs["input_ids"].to('cuda:0')

        generation_config = GenerationConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=40,
            num_beams=1,
            repetition_penalty=1.0,
            max_memory=self.max_memory,
        )
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                # input_ids=input_ids.to('cuda:0'),

                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                max_new_tokens=self.max_token,
                repetition_penalty=float(1.0),
            )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s, skip_special_tokens=True)
        output = output.split("Assistant:")[-1].strip()
        print("output:", output)
        print("====================")
        self.history = self.history + [[now_input, output]]
        return output
    
    def generate_prompt(self,instruction):
        return f'</s>Human:{instruction} </s>Assistant: '        

    def load_model(self,
                   model_name_or_path: str = "THUDM/chatglm-6b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            load_in_8bit=False,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map='auto',
            trust_remote_code=True,
            ).half().cuda()
        # self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", trust_remote_code=True)
        self.model = self.model.eval()

    def auto_configure_device_map(self, num_gpus: int) -> Dict[str, int]:
        # transformer.word_embeddings 占用1层
        # transformer.final_layernorm 和 lm_head 占用1层
        # transformer.layers 占用 28 层
        # 总共30层分配到num_gpus张卡上
        num_trans_layers = 28
        per_gpu_layers = 30 / num_gpus

        # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
        # windows下 model.device 会被设置成 transformer.word_embeddings.device
        # linux下 model.device 会被设置成 lm_head.device
        # 在调用chat或者stream_chat时,input_ids会被放到model.device上
        # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
        # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
        device_map = {'transformer.word_embeddings': 0,
                      'transformer.final_layernorm': 0, 'lm_head': 0}

        used = 2
        gpu_target = 0
        for i in range(num_trans_layers):
            if used >= per_gpu_layers:
                gpu_target += 1
                used = 0
            assert gpu_target < num_gpus
            device_map[f'transformer.layers.{i}'] = gpu_target
            used += 1

        return device_map

    def load_model_on_gpus(self, model_name_or_path: Union[str, os.PathLike], num_gpus: int = 2,
                           multi_gpu_model_cache_dir: Union[str, os.PathLike] = "./temp_model_dir",
                           ):
        # https://github.com/THUDM/ChatGLM-6B/issues/200
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, )
        self.model = self.model.eval()

        device_map = self.auto_configure_device_map(num_gpus)
        try:
            self.model = load_checkpoint_and_dispatch(
                self.model, model_name_or_path, device_map=device_map, offload_folder="offload",
                offload_state_dict=True).half()
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True
            )
        except ValueError:
            # index.json not found
            print(f"index.json not found, auto fixing and saving model to {multi_gpu_model_cache_dir} ...")

            assert multi_gpu_model_cache_dir is not None, "using auto fix, cache_dir must not be None"
            self.model.save_pretrained(multi_gpu_model_cache_dir, max_shard_size='2GB')
            self.model = load_checkpoint_and_dispatch(
                self.model, multi_gpu_model_cache_dir, device_map=device_map,
                offload_folder="offload", offload_state_dict=True).half()
            self.tokenizer = AutoTokenizer.from_pretrained(
                multi_gpu_model_cache_dir,
                trust_remote_code=True
            )
            print(f"loading model successfully, you should use checkpoint_path={multi_gpu_model_cache_dir} next time")