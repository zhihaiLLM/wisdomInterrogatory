import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import re
import torch
import gradio as gr
from clc.langchain_application import LangChainApplication, torch_gc
from transformers import StoppingCriteriaList, StoppingCriteriaList
from clc.callbacks import Iteratorize, Stream
from clc.matching import key_words_match_intention, key_words_match_knowledge
from langchain.schema import Document
# 调试使用
# os.chdir("../../../")

class LangChainCFG:
    llm_model_name = 'luwen_baichuan/output/zju_model_0813_100k'  # 本地模型文件 or huggingface远程仓库
    embedding_model_name = 'app/langchain_demo/model/text2vec'  # 检索模型文件 or huggingface远程仓库
    vector_store_path = 'app/langchain_demo/data/cache/legal_articles'
    kg_vector_stores = {
        '法律法条': 'app/langchain_demo/data/cache/legal_articles',
        '法律书籍': 'app/langchain_demo/data/cache/legal_books',
        '法律文书模板':'app/langchain_demo/data/cache/legal_templates',
        '法律案例': 'app/langchain_demo/data/cache/legal_cases',
        '法律考试': 'app/langchain_demo/data/cache/judicialExamination',
        '日常法律问答': 'app/langchain_demo/data/cache/legal_QA',
    }  

config = LangChainCFG()
application = LangChainApplication(config)

def clear_session():
    return '', None, ""

def predict(input,
            kg_names=None,
            history=None,
            intention_reg=None,
            **kwargs):
    max_length=1024
    top_k = 1
    application.llm_service.max_token = max_length
    # print(input)
    if history == None:
        history = []
    search_text = ''

    now_input = input
    eos_token_ids = [application.llm_service.tokenizer.eos_token_id]
    application.llm_service.history = history[-5:]
    max_memory = 4096 - max_length

    if intention_reg==["意图识别"]:
        auto_kg_names = key_words_match_intention(input)
        if len(auto_kg_names)==0:
            search_text += "意图识别没有匹配到知识库。\n\n"
        else:
            match_kg_names = "、".join(list(auto_kg_names))
            search_text += "意图识别匹配到知识库是："+match_kg_names+"。\n\n"
        kg_names = list(set(kg_names) | auto_kg_names)

    kb_based = True if len(kg_names) != 0 else False

    if len(history) != 0:
        input = "".join(["</s>Human:" + i[0] + " </s>Assistant: " + i[1] for i in application.llm_service.history]) + \
        "</s>Human:" + input
        input = input[len("</s>Human:"):]
    if len(input) > max_memory:
        input = input[-max_memory:]

    if kb_based:
        related_docs_with_score_seq = []
        for kg_name in kg_names:
            if kg_name=="法律法条":
                related_article = key_words_match_knowledge(application.all_articles, application.choices, now_input)
                if related_article:
                    kg_matches = [(Document(page_content=related_article[0], metadata={"value": related_article[1]}),0)]
                else:
                    application.source_service.load_vector_store(application.config.kg_vector_stores[kg_name])
                    kg_matches = application.source_service.vector_store.similarity_search_with_score(input, k=top_k)
            else:
                application.source_service.load_vector_store(application.config.kg_vector_stores[kg_name])
                kg_matches = application.source_service.vector_store.similarity_search_with_score(input, k=top_k)
            related_docs_with_score_seq.append(kg_matches)
        related_docs_with_score = related_docs_with_score_seq
        if len(related_docs_with_score) > 0:
            input, context_with_score = application.generate_prompt(related_docs_with_score, input,kg_names)
        search_text += context_with_score
    torch_gc()

    print("histroy in call: ", history)
    prompt = f'</s>Human:{input} </s>Assistant: ' 
    print("prompt: ",prompt)
    inputs = application.llm_service.tokenizer(prompt, return_tensors="pt").to('cuda')
    stopping_criteria = StoppingCriteriaList()

    kwargs['inputs'] = inputs
    kwargs['max_new_tokens'] = max_length
    kwargs['repetition_penalty'] = float(1.2)
    kwargs['stopping_criteria'] = stopping_criteria
    history.append((now_input, ""))

    def generate_with_callback(callback=None, **kwargs):
        kwargs['stopping_criteria'].append(Stream(callback_func=callback))
        with torch.no_grad():
            application.llm_service.model.generate(**kwargs['inputs'], 
                                                    max_new_tokens=kwargs['max_new_tokens'], 
                                                    repetition_penalty=kwargs['repetition_penalty'],
                                                    stopping_criteria=kwargs["stopping_criteria"])

    def generate_with_streaming(**kwargs):
        return Iteratorize(generate_with_callback, kwargs, callback=None)

    with generate_with_streaming(**kwargs) as generator:
        for output in generator:
            last = output[-1]
            output = application.llm_service.tokenizer.decode(output, skip_special_tokens=True)
            pattern = r"\n{5,}$"
            pattern2 = r"\s{5,}$"
            origin_output = output
            output = output.split("Assistant:")[-1].strip()
            history[-1] = (now_input, output)
            yield "", history, history, search_text
            if last in eos_token_ids or re.search(pattern, origin_output) or re.search(pattern2, origin_output):
                break

with gr.Blocks() as demo: 
    state = gr.State()
    with gr.Row():
        with gr.Column(scale=1.5):
            github_banner_path = 'https://raw.githubusercontent.com/LIANG-star177/chatgptapi/master/logo.png'
            gr.HTML(f'<p align="center"><a href="https://github.com/LIANG-star177/chatgptapi/blob/master/logo.png"><img src={github_banner_path} height="100" width="200"/></a></p>')
            with gr.Row():       
                intention_reg = gr.CheckboxGroup(["意图识别"],
                        label="自动选择知识库",
                        value=None,
                        interactive=True)
            with gr.Row():
                kg_names = gr.CheckboxGroup(list(config.kg_vector_stores.keys()),
                                label="手动选择知识库",
                                value=None,
                                interactive=True).style(height=200)
            with gr.Row():
                search = gr.Textbox(label='知识库检索结果')

            with gr.Row():
                gr.Markdown("""Powered by 浙江大学 阿里巴巴达摩院 华院计算 魔搭社区""")
            with gr.Row():
                gr.Markdown("""免责声明：本模型仅供学术研究之目的而提供，不保证结果的准确性、完整性或适用性。在使用模型生成的内容时，您应自行判断其适用性，并自担风险。""")
        with gr.Column(scale=4):
            with gr.Row():
                chatbot = gr.Chatbot(label='智海-录问').style(height=500)            
            with gr.Row():
                message = gr.Textbox(label='请输入问题')            
            with gr.Row():
                clear_history = gr.Button("🧹 清除历史对话")
                send = gr.Button("🚀 发送")

        send.click(predict,
                   inputs=[
                    message,
                    kg_names,
                    state,
                    intention_reg,
                   ],
                   outputs=[message, chatbot, state, search],
                   show_progress=True)

        clear_history.click(fn=clear_session,
                            inputs=[],
                            outputs=[chatbot, state, search],
                            queue=False)

        message.submit(predict,
                       inputs=[
                        message,
                        kg_names,
                        state,
                        intention_reg,
                       ],
                       outputs=[message, chatbot, state, search],
                       show_progress=True)

demo.queue(concurrency_count=2).launch(
    server_name='0.0.0.0',
    server_port=7888,
    share=True,
    enable_queue=True,
    inbrowser=True,
)
