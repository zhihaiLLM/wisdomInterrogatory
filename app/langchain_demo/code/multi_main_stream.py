import os
import json
import re
import shutil
import torch
from app_modules.overwrites import postprocess
from app_modules.presets import *
from clc.langchain_application import LangChainApplication, torch_gc
from transformers import StoppingCriteriaList
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextIteratorStreamer, StoppingCriteriaList
from clc.callbacks import Iteratorize, Stream, _SentinelTokenStoppingCriteria, clear_torch_cache

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
server_port = 8889
device1 = torch.device("cuda:0")  # 使用第一张GPU卡
device2 = torch.device("cuda:1")  # 使用第二张GPU卡
# 修改成自己的配置！！！
class LangChainCFG:
    llm_model_name = '/root/data1/luwen/luwen_baichuan/output/zju_model_0801_80k_new'  # 本地模型文件 or huggingface远程仓库
    llm_model_name2 = '/root/data1/luwen/luwen_baichuan/output/zju_model_0801_80k_new'
    embedding_model_name = '/root/data1/luwen/app/langchain_demo/model/text2vec/text2vec_large'  # 检索模型文件 or huggingface远程仓库
    vector_store_path = '/root/data1/luwen/app/langchain_demo/data/cache'
    kg_vector_stores = {
        '法律法条': '/root/data1/luwen/app/langchain_demo/data/cache/legal_article',
        '法律书籍': '/root/data1/luwen/app/langchain_demo/data/cache/legal_book',
        '法律文书模版':'/root/data1/luwen/app/langchain_demo/data/cache/legal_template',
        '法律案例': '/root/data1/luwen/app/langchain_demo/data/cache/legal_case',
        '法律考试': '/root/data1/luwen/app/langchain_demo/data/cache/judicialExamination',
    }  # 可以替换成自己的知识库，如果没有需要设置为None
    # kg_vector_stores=None
    patterns = ['模型问答', '知识库问答']  #
    # n_gpus=1
    device = {"dev1":device1,"dev2":device2}



config = LangChainCFG()
application = LangChainApplication(config)

# application.source_service.init_source_vector()

def get_file_list():
    if not os.path.exists("docs"):
        return []
    return [f for f in os.listdir("docs")]


file_list = get_file_list()


def upload_file(file):
    if not os.path.exists("docs"):
        os.mkdir("docs")
    filename = os.path.basename(file.name)
    shutil.move(file.name, "docs/" + filename)
    # file_list首位插入新上传的文件
    file_list.insert(0, filename)
    application.source_service.add_document("docs/" + filename)
    return gr.Dropdown.update(choices=file_list, value=filename)


def set_knowledge(kg_names, history):
    kg_print_out = ""
    try:
        for kg_name in kg_names:
            application.source_service.load_vector_store(config.kg_vector_stores[kg_name])
            kg_print_out = kg_print_out + "  " + kg_name
        msg_status = f'{kg_print_out}知识库已成功加载'
    except Exception as e:
        print(e)
        msg_status = f'{kg_name}知识库未成功加载'
    return history + [[None, msg_status]]


def clear_session():
    return '', None, ""

def key_words_match(input):
    kg_names = set()
    with open("/root/data1/luwen/app/langchain_demo/data/cache/intention_reg.json", "r") as f:
        dic = json.load(f)
        for key,val in dic.items():
            for el in val:
                if el in input:
                    kg_names.add(key)
    return kg_names


def predict(input,
            # chatbot,
            # large_language_model="zju-bc",
            # embedding_model="text2vec",
            kg_names=None,
            history=None,
            # max_length=1024,
            # top_k = 1,
            intention_reg=None,
            **kwargs):
    # print(large_language_model, embedding_model)
    large_language_model="zju-bc"
    embedding_model="text2vec"
    max_length=1024
    top_k = 1
    if large_language_model=="zju-lm":
        application.llm_service.tokenizer = application.tokenizer2
        application.llm_service.model = application.model2
    application.llm_service.max_token = max_length
    # print(input)
    if history == None:
        history = []
    search_text = ''

    now_input = input
    # chatbot.append((input, ""))
    # history.append((now_input, ""))
    eos_token_ids = [application.llm_service.tokenizer.eos_token_id]
    application.llm_service.history = history[-5:]
    max_memory = 4096 - max_length

    if intention_reg==["意图识别"]:
        auto_kg_names = key_words_match(input)
        # if len(auto_kg_names)==0:
        #     # search_text = "没有找到合适的知识库！\n"
        #     kg_names = []
        # else:
        #     kg_names = list(auto_kg_names)
        kg_names = list(set(kg_names) | auto_kg_names)

    kb_based = True if len(kg_names) != 0 else False

    if len(history) != 0:
        if large_language_model=="zju-bc":
            input = "".join(["</s>Human:\n" + i[0] +"\n" + "</s>Assistant:\n" + i[1] + "\n"for i in application.llm_service.history]) + \
            "</s>Human:\n" + input
            input = input[len("</s>Human:\n"):]
        else:
            input = "".join(["### Instruction:\n" + i[0] +"\n" + "### Response: " + i[1] + "\n" for i in application.llm_service.history]) + \
            "### Instruction:\n" + input
            input = input[len("### Instruction:\n"):]
    if len(input) > max_memory:
        input = input[-max_memory:]

    if kb_based:
        if len(kg_names) == 0:
            kg_names = ["刑法法条"]
        related_docs_with_score_seq = []
        for kg_name in kg_names:
            application.source_service.load_vector_store(application.config.kg_vector_stores[kg_name])
            related_docs_with_score_seq.append(application.source_service.vector_store.similarity_search_with_score(input, k=top_k))
        related_docs_with_score = related_docs_with_score_seq
        if len(related_docs_with_score) > 0:
            input, context_with_score = application.generate_prompt(related_docs_with_score, input, large_language_model,kg_names)
        search_text += context_with_score
    torch_gc()

    print("histroy in call: ", history)
    prompt = application.llm_service.generate_prompt(input, kb_based, large_language_model)
    print("prompt: ",prompt)
    inputs = application.llm_service.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    stopping_criteria = StoppingCriteriaList()

    kwargs['input_ids'] = input_ids
    kwargs['temperature'] = 0.1
    kwargs['top_p'] = 0.9
    kwargs['return_dict_in_generate'] = True
    kwargs['max_new_tokens'] = max_length
    kwargs['repetition_penalty'] = float(1.2)
    kwargs['stopping_criteria'] = stopping_criteria
    history.append((now_input, ""))

    def generate_with_callback(callback=None, **kwargs):
        kwargs['stopping_criteria'].append(Stream(callback_func=callback))
        # clear_torch_cache(False)
        generation_config = GenerationConfig(
            temperature=kwargs['temperature'],
            top_p=kwargs['top_p'],
            top_k=40,
            num_beams=1,
            repetition_penalty=1.2,
            max_memory=max_memory,
        )
        with torch.no_grad():
            application.llm_service.model.generate(
                input_ids=kwargs['input_ids'],
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                max_new_tokens=kwargs['max_new_tokens'],
                stopping_criteria=kwargs["stopping_criteria"]
            )

    def generate_with_streaming(**kwargs):
        return Iteratorize(generate_with_callback, kwargs, callback=None)

    with generate_with_streaming(**kwargs) as generator:
        for output in generator:
            last = output[-1]
            output = application.llm_service.tokenizer.decode(output, skip_special_tokens=True)
            pattern = r"\n{5,}$"
            origin_output = output
            if large_language_model=="zju-bc":
                output = output.split("Assistant:")[-1].strip()
            else:
                output = output.split("### Response:")[-1].strip()
            history[-1] = (now_input, output)
            yield "", history, history, search_text
            if last in eos_token_ids or re.search(pattern, origin_output):
                break


with open("/root/data1/luwen/app/langchain_demo/code/assets/custom.css", "r", encoding="utf-8") as f:
    customCSS = f.read()
# with gr.Blocks(css=customCSS, theme=small_and_beautiful_theme) as demo:
with gr.Blocks() as demo:    
    gr.Markdown("""<h1><center>智海-录问</center></h1>
        <center><font size=3>
        </center></font>
        """)
    state = gr.State()

    # with gr.Row():
        # intention_reg = gr.Radio(
        #     [
        #         '是',
        #     ],
        #     value=NONE,
        #     interactive=True)

    with gr.Row():
        with gr.Column(scale=1):
            # with gr.Row():
            #     # gr.Label("问答参数配置", css="font-size: 12px; padding: 0;")
            #     embedding_model = gr.Dropdown([
            #         "text2vec"
            #     ],
            #         label="Embedding model",
            #         value="text2vec")

            #     large_language_model = gr.Dropdown(
            #         [
            #             "zju-bc",
            #             "zju-lm",
            #         ],
            #         label="large language model",
            #         value="zju-bc")
            #     max_length = gr.Slider(
            #             0, 4096, value=1024, step=1.0, label="Maximum length", interactive=True)

            # top_k = 1
            # top_k = gr.Slider(1,
            #                   20,
            #                   value=1,
            #                   step=1,
            #                   label="检索top-k文档",
            #                   interactive=True)

            # use_web = gr.Radio(["使用", "不使用"], label="web search",
            #                    info="是否使用网络搜索，使用时确保网络通常",
            #                    value="不使用"
            #                    )
            # use_web = "不使用"
            with gr.Row():       
                intention_reg = gr.CheckboxGroup(["意图识别"],
                        label="自动选择知识库",
                        value=None,
                        interactive=True)
            with gr.Row():
                # gr.Label("知识库辅助问答", css="font-size: 12px; padding: 0;")

                kg_names = gr.CheckboxGroup(list(config.kg_vector_stores.keys()),
                                label="手动选择知识库",
                                value=None,
                                # info="使用知识库问答，请加载知识库",
                                interactive=True).style(height=200)
            with gr.Row():
                search = gr.Textbox(label='知识库检索结果')

            with gr.Row():
                gr.Markdown("""powered by 浙江大学 阿里巴巴达摩院 华院计算""")

            # file = gr.File(label="将文件上传到知识库库，内容要尽量匹配",
            #                visible=True,
            #                file_types=['.txt', '.md', '.docx', '.pdf']
            #                )

        with gr.Column(scale=4):
            with gr.Row():
                chatbot = gr.Chatbot(label='智海-录问').style(height=500)            
            with gr.Row():
                message = gr.Textbox(label='请输入问题')            
            with gr.Row():
                clear_history = gr.Button("🧹 清除历史对话")
                send = gr.Button("🚀 发送")
            # with gr.Row():
            # #     gr.Markdown("""提醒：<br>
            # #                             司法大模型-ZJU是基于深度学习技术构建的，它可以提供有价值的法律建议和解释，但不应视为法律专家的替代品。在重要的法律事务中，建议您咨询专业的法律顾问或律师。 <br>
            # #                             """)

            #     search = gr.Textbox(label='知识库检索结果')
        
        # ============= 触发动作=============
        # file.upload(upload_file,
        #             inputs=file,
        #             outputs=None)
        # set_kg_btn.click(
        #     set_knowledge,
        #     show_progress=True,
        #     inputs=[kg_names, chatbot],
        #     outputs=chatbot
        # )
        # 发送按钮 提交
        send.click(predict,
                   inputs=[
                       message,
                    #    chatbot,
                    #    large_language_model,
                    #    embedding_model,
                       kg_names,
                       state,
                    #    max_length,
                    intention_reg,
                   ],
                   outputs=[message, chatbot, state, search],
                   show_progress=True)

        # 清空历史对话按钮 提交
        clear_history.click(fn=clear_session,
                            inputs=[],
                            outputs=[chatbot, state, search],
                            queue=False)

        # 输入框 回车
        message.submit(predict,
                       inputs=[
                           message,
                        #    large_language_model,
                        #    embedding_model,
                           kg_names,
                           state,
                        #    max_length
                        intention_reg,
                       ],
                       outputs=[message, chatbot, state, search],
                       show_progress=True)

demo.queue(concurrency_count=2).launch(
    server_name='0.0.0.0',
    server_port=server_port,
    share=True,
    # show_error=True,
    enable_queue=True,
    inbrowser=True,
)
