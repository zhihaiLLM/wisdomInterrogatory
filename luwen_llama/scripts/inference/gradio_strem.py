import sys
import gradio as gr
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default="/root/data1/luwen/luwen_llama/mymodel/alpaca_plus_13b_sft298k_hf", type=str)
parser.add_argument('--lora_model', default=None, type=str,help="If None, perform inference on the base model")
parser.add_argument('--tokenizer_path',default=None,type=str)
parser.add_argument('--gpus', default="1", type=str)
parser.add_argument('--share', default=True, help='share gradio domain name')
parser.add_argument('--load_in_8bit',action='store_true', help='use 8 bit model')
parser.add_argument('--only_cpu',action='store_true',help='only use CPU for inference')
args = parser.parse_args()
share = args.share
load_in_8bit = args.load_in_8bit
if args.only_cpu is True:
    args.gpus = ""
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig, TextIteratorStreamer, StoppingCriteriaList
from peft import PeftModel
from callbacks import Iteratorize, Stream, _SentinelTokenStoppingCriteria, clear_torch_cache


generation_config = dict(
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.1,
    max_new_tokens=400
    )
load_type = torch.float16
if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device('cpu')
if args.tokenizer_path is None:
    args.tokenizer_path = args.lora_model
    if args.lora_model is None:
        args.tokenizer_path = args.base_model
tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
eos_token_ids = [tokenizer.eos_token_id]


base_model = LlamaForCausalLM.from_pretrained(
    args.base_model, 
    load_in_8bit=load_in_8bit,
    torch_dtype=load_type,
    low_cpu_mem_usage=True,
    device_map='auto',
    )

model_vocab_size = base_model.get_input_embeddings().weight.size(0)
tokenzier_vocab_size = len(tokenizer)
print(f"Vocab of the base model: {model_vocab_size}")
print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
if model_vocab_size!=tokenzier_vocab_size:
    assert tokenzier_vocab_size > model_vocab_size
    print("Resize model embeddings to fit tokenizer")
    base_model.resize_token_embeddings(tokenzier_vocab_size)
if args.lora_model is not None:
    print("loading peft model")
    model = PeftModel.from_pretrained(base_model, args.lora_model,torch_dtype=load_type,device_map='auto',)
else:
    model = base_model

if device==torch.device('cpu'):
    model.float()

model.eval()

def reset_user_input():
    return gr.update(value='')

def reset_state():
    return [], []

def generate_prompt(instruction):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
    
### Instruction:
{instruction}

### Response: """

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

def predict(
    input,
    chatbot,
    history,
    max_new_tokens=128,
    top_p=0.75,
    temperature=0.1,
    top_k=40,
    num_beams=4,
    repetition_penalty=1.0,
    max_memory=256,
    **kwargs,
):
    now_input = input
    chatbot.append((input, ""))
    history = history or []
    if len(history) != 0:
        input = "".join(["### Instruction:\n" + i[0] +"\n\n" + "### Response: " + i[1] + "\n\n" for i in history]) + \
        "### Instruction:\n" + input
        input = input[len("### Instruction:\n"):]
        if len(input) > max_memory:
            input = input[-max_memory:]
    prompt = generate_prompt(input)
    print("prompt:", prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    stopping_criteria_list = StoppingCriteriaList()

    # generation_config = GenerationConfig(
    #     temperature=temperature,
    #     top_p=top_p,
    #     top_k=top_k,
    #     num_beams=num_beams,
    #     **kwargs,
    # )
    kwargs['temperature'] = temperature
    kwargs['top_p'] = top_p
    kwargs['top_k'] = top_k
    kwargs['num_beams'] = num_beams
    kwargs['return_dict_in_generate'] = True
    kwargs['input_ids'] = input_ids
    kwargs['output_scores'] = False
    kwargs['max_new_tokens'] = max_new_tokens
    kwargs['repetition_penalty'] = float(repetition_penalty)
    kwargs['stopping_criteria'] = stopping_criteria_list
    # with torch.no_grad():
    #     generation_output = model.generate(
    #         input_ids=input_ids,
    #         generation_config=generation_config,
    #         return_dict_in_generate=True,
    #         output_scores=False,
    #         max_new_tokens=max_new_tokens,
    #         repetition_penalty=float(repetition_penalty),
    #     )
    # s = generation_output.sequences[0]
    # output = tokenizer.decode(s, skip_special_tokens=True)
    # output = output.split("### Response:")[-1].strip()
    # print("output:", output)
    # print("====================")
    # history.append((now_input, output))
    # chatbot[-1] = (now_input, output)
    # return chatbot, history
    history.append((now_input, ""))

    def generate_with_callback(callback=None, **kwargs):
        kwargs['stopping_criteria'].append(Stream(callback_func=callback))
        clear_torch_cache(args.only_cpu)
        with torch.no_grad():
            model.generate(**kwargs)

    def generate_with_streaming(**kwargs):
        return Iteratorize(generate_with_callback, kwargs, callback=None)

    with generate_with_streaming(**kwargs) as generator:
        for output in generator:
            last = output[-1]
            output = tokenizer.decode(output, skip_special_tokens=True)
            output = output.split("### Response:")[-1].strip()

            # print(output)
            history[-1] = (now_input, output)
            chatbot[-1] = (now_input, output)
            yield chatbot, history
            if last in eos_token_ids:
                break

with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">Chinese LLaMA & Alpaca LLM</h1>""")
    current_file_path = os.path.abspath(os.path.dirname(__file__))
    gr.Image(f'{current_file_path}/../../pics/banner.png', label = 'Chinese LLaMA & Alpaca LLM')
    gr.Markdown("> 为了促进大模型在中文NLP社区的开放研究，本项目开源了中文LLaMA模型和指令精调的Alpaca大模型。这些模型在原版LLaMA的基础上扩充了中文词表并使用了中文数据进行二次预训练，进一步提升了中文基础语义理解能力。同时，中文Alpaca模型进一步使用了中文指令数据进行精调，显著提升了模型对指令的理解和执行能力")
    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(
                0, 4096, value=128, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01,
                              label="Top P", interactive=True)
            temperature = gr.Slider(
                0, 1, value=0.7, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])  # (message, bot_message)

    submitBtn.click(predict, [user_input, chatbot, history, max_length, top_p, temperature], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)
demo.queue().launch(share=share, inbrowser=True, server_name = '0.0.0.0', server_port=8080)
