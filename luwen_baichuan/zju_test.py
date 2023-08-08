import sys

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(sys.argv[1], device_map="auto", trust_remote_code=True)
inputs = tokenizer(f'</s>Human: hello!</s>Assistant: ', return_tensors='pt')
inputs = inputs.to('cuda:0')
pred = model.generate(**inputs, max_new_tokens=300, repetition_penalty=1)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
