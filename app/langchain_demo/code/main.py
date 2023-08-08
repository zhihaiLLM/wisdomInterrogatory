import os
import shutil
import torch
from app_modules.overwrites import postprocess
from app_modules.presets import *
from clc.langchain_application import LangChainApplication
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device1 = torch.device("cuda:0")  # 使用第一张GPU卡
device2 = torch.device("cuda:1")  # 使用第二张GPU卡
# 修改成自己的配置！！！
class LangChainCFG:
    llm_model_name = '/root/data1/luwen/luwen_baichuan/output/zju_model_0710_100k_wenshu30k'  # 本地模型文件 or huggingface远程仓库
    llm_model_name2 = '/root/data1/luwen/luwen_llama/mymodel/alpaca_plus_13b_sft298k_hf'
    embedding_model_name = '/root/data1/luwen/luwen_llama/mymodel/text2vec_large'  # 检索模型文件 or huggingface远程仓库
    vector_store_path = 'scripts/langchain_demo/cache2/'
    docs_path = 'scripts/langchain_demo/test_docs'
    kg_vector_stores = {
        '刑法法条': 'scripts/langchain_demo/cache2/legal_article',
        '刑法书籍': 'scripts/langchain_demo/cache2/legal_book',
        '法律文书模版':'scripts/langchain_demo/cache2/legal_template',
        '刑法案例': 'scripts/langchain_demo/cache2/legal_case',
        # '初始化': 'scripts/langchain_demo/cache',
    }  # 可以替换成自己的知识库，如果没有需要设置为None
    # kg_vector_stores=None
    patterns = ['模型问答', '知识库问答']  #
    # n_gpus=1
    device = {"dev1":device1,"dev2":device2}



config = LangChainCFG()
application = LangChainApplication(config)

application.source_service.init_source_vector()

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
    return '', None


def predict(input,
            large_language_model,
            embedding_model,
            top_k,
            # use_web,
            use_pattern,
            kg_names,
            history=None,
            max_length=None):
    # print(large_language_model, embedding_model)
    if large_language_model=="zju-lm":
        application.llm_service.tokenizer = application.tokenizer2
        application.llm_service.model = application.model2
    application.llm_service.max_token = max_length
    # print(input)
    if history == None:
        history = []
    use_web = "不使用"
    if use_web == '使用':
        web_content = application.source_service.search_web(query=input)
    else:
        web_content = ''
    search_text = ''
    if use_pattern == '模型问答':
        result = application.get_llm_answer(query=input, web_content=web_content,chat_history=history, model_name=large_language_model)
        history.append((input, result))
        search_text += web_content
        return '', history, history, search_text

    else:
        result, context_with_score = application.get_knowledge_based_answer(
            query=input,
            history_len=5,
            temperature=0.1,
            top_p=0.9,
            top_k=top_k,
            web_content=web_content,
            chat_history=history,
            kg_names = kg_names,
            model_name = large_language_model
        )
        history.append((input, result))
        search_text += context_with_score
        # history.append((input, resp['result']))
        # for idx, source in enumerate(resp['source_documents'][:4]):
        #     sep = f'----------【搜索结果{idx + 1}：】---------------\n'
        #     search_text += f'{sep}\n{source.page_content}\n\n'
        # print(search_text)
        # search_text += "----------【网络检索内容】-----------\n"
        # search_text += web_content
        return '', history, history, search_text


with open("scripts/langchain_demo/assets/custom.css", "r", encoding="utf-8") as f:
    customCSS = f.read()
# with gr.Blocks(css=customCSS, theme=small_and_beautiful_theme) as demo:
with gr.Blocks() as demo:    
    gr.Markdown("""<h1><center>智海-录问</center></h1>
        <center><font size=3>
        </center></font>
        """)
    state = gr.State()

    with gr.Row():
        with gr.Column(scale=1):

            top_k = gr.Slider(1,
                              20,
                              value=4,
                              step=1,
                              label="检索top-k文档",
                              interactive=True)

            # use_web = gr.Radio(["使用", "不使用"], label="web search",
            #                    info="是否使用网络搜索，使用时确保网络通常",
            #                    value="不使用"
            #                    )
            # use_web = "不使用"
            use_pattern = gr.Radio(
                [
                    '模型问答',
                    '知识库问答',
                ],
                label="模式",
                value='模型问答',
                interactive=True)

            kg_names = gr.CheckboxGroup(list(config.kg_vector_stores.keys()),
                               label="知识库",
                               value=None,
                               info="使用知识库问答，请加载知识库",
                               interactive=True).style(height=200)
            set_kg_btn = gr.Button("加载知识库")
            with gr.Row():
                gr.Markdown("""提醒：<br>
                                        智海-录问是基于深度学习技术构建的，它可以提供有价值的法律建议和解释，但不应视为法律专家的替代品。在重要的法律事务中，建议您咨询专业的法律顾问或律师。 <br>
                                        """)

            # file = gr.File(label="将文件上传到知识库库，内容要尽量匹配",
            #                visible=True,
            #                file_types=['.txt', '.md', '.docx', '.pdf']
            #                )

        with gr.Column(scale=4):
            with gr.Row():
                chatbot = gr.Chatbot(label='智海-录问').style(height=300)
            with gr.Row():
                message = gr.Textbox(label='请输入问题')
            with gr.Row():
                clear_history = gr.Button("🧹 清除历史对话")
                send = gr.Button("🚀 发送")
            # with gr.Row():
            #     gr.Markdown("""提醒：<br>
            #                             司法大模型-ZJU是基于深度学习技术构建的，它可以提供有价值的法律建议和解释，但不应视为法律专家的替代品。在重要的法律事务中，建议您咨询专业的法律顾问或律师。 <br>
            #                             """)
        with gr.Column(scale=2):
            embedding_model = gr.Dropdown([
                "text2vec-large"
            ],
                label="Embedding model",
                value="text2vec-large")

            large_language_model = gr.Dropdown(
                [
                    "zju-bc",
                    "zju-lm",
                ],
                label="large language model",
                value="zju-bc")
            max_length = gr.Slider(
                    0, 4096, value=1024, step=1.0, label="Maximum length", interactive=True)
        # with gr.Column(scale=2):
            search = gr.Textbox(label='搜索结果')
        
        # ============= 触发动作=============
        # file.upload(upload_file,
        #             inputs=file,
        #             outputs=None)
        set_kg_btn.click(
            set_knowledge,
            show_progress=True,
            inputs=[kg_names, chatbot],
            outputs=chatbot
        )
        # 发送按钮 提交
        send.click(predict,
                   inputs=[
                       message,
                       large_language_model,
                       embedding_model,
                       top_k,
                    #    use_web,
                       use_pattern,
                       kg_names,
                       state,
                       max_length,
                   ],
                   outputs=[message, chatbot, state, search])

        # 清空历史对话按钮 提交
        clear_history.click(fn=clear_session,
                            inputs=[],
                            outputs=[chatbot, state],
                            queue=False)

        # 输入框 回车
        message.submit(predict,
                       inputs=[
                           message,
                           large_language_model,
                           embedding_model,
                           top_k,
                        #    use_web,
                           use_pattern,
                           kg_names,
                           state,
                           max_length
                       ],
                       outputs=[message, chatbot, state, search])

demo.queue(concurrency_count=2).launch(
    server_name='0.0.0.0',
    # server_port=8888,
    share=False,
    show_error=True,
    debug=True,
    enable_queue=True,
    inbrowser=True,
)
