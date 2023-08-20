from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_path = "output/zju_model_0813_100k"

def generate_response(prompt):
    torch.cuda.empty_cache()
    inputs = tokenizer(f'</s>Human:{prompt} </s>Assistant: ', return_tensors='pt')
    inputs = inputs.to('cuda')
    with torch.no_grad():
        pred = model.generate(**inputs, max_new_tokens=800, repetition_penalty=1.2)
    response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
    return response.split("Assistant: ")[1]

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).half()
prompt = "如果喝了两斤白酒后开车，会有什么后果？"

resp = generate_response(prompt)
print(resp)