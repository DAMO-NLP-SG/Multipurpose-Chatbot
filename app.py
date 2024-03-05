# Copyright: DAMO Academy, Alibaba Group
# By Xuan Phi Nguyen at DAMO Academy, Alibaba Group

# Description:
"""
Demo script to launch Language chat model for Southeast Asian Languages
"""


import os
from gradio.themes import ThemeClass as Theme
import numpy as np
import argparse
# import torch
import gradio as gr
from typing import Any, Iterator
from typing import Iterator, List, Optional, Tuple
import filelock
import glob
import json
import time
from gradio.routes import Request
from gradio.utils import SyncToAsyncIterator, async_iteration
from gradio.helpers import special_args
import anyio
from typing import AsyncGenerator, Callable, Literal, Union, cast

from gradio_client.documentation import document, set_documentation_group

from typing import List, Optional, Union, Dict, Tuple
from tqdm.auto import tqdm
from huggingface_hub import snapshot_download
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from gradio.components import Button, Component
from gradio.events import Dependency, EventListenerMethod

from multipurpose_chatbot.demos.base_demo import CustomTabbedInterface



# @@ environments ================

# gradio config
PORT = int(os.environ.get("PORT", "7860"))
PROXY = os.environ.get("PROXY", "").strip()

MODEL_PATH = os.environ.get("MODEL_PATH", "./seal-13b-chat-a")
MODEL_NAME = os.environ.get("MODEL_NAME", "Cool-Chatbot")
# whether to enable to popup accept user
ENABLE_AGREE_POPUP = bool(int(os.environ.get("ENABLE_AGREE_POPUP", "0")))

BACKEND = os.environ.get("BACKEND", "mlx")


allowed_paths = []
demo = None

MODEL_TITLE = "<h1>Multi-Purpose Chatbot</h1>"

MODEL_DESC = f"""
<div style='display:flex; gap: 0.25rem; '>
<a href='https://github.com/DAMO-NLP-SG/Multipurpose-Chatbot'><img src='https://img.shields.io/badge/Github-Code-success'></a>
</div>
<span style="font-size: larger">
A multi-purpose helpful assistant with multiple functionalities (Chat, Freeform, batch, RAG...).
</span>
""".strip()



path_markdown = """
<h4>Model Name: {model_path}</h4>
"""


cite_markdown = """
## Citation
If you find our project useful, hope you can star our repo and cite our paper as follows:
```
@article{damonlpsg2023seallm,
  author = {Xuan-Phi Nguyen*, Wenxuan Zhang*, Xin Li*, Mahani Aljunied*, Zhiqiang Hu, Chenhui Shen^, Yew Ken Chia^, Xingxuan Li, Jianyu Wang, Qingyu Tan, Liying Cheng, Guanzheng Chen, Yue Deng, Sen Yang, Chaoqun Liu, Hang Zhang, Lidong Bing},
  title = {SeaLLMs - Large Language Models for Southeast Asia},
  year = 2023,
}
```
"""

set_documentation_group("component")


# MODEL_ENGINE = None
# RAG_EMBED = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'trust_remote_code':True})
RAG_CURRENT_FILE = None
RAG_CURRENT_VECTORSTORE = None



def launch_demo():
    global demo, MODEL_ENGINE
    model_desc = MODEL_DESC
    model_path = MODEL_PATH
    # model_title = MODEL_TITLE
    # hf_model_name = HF_MODEL_NAME
    # dtype = DTYPE
    # sys_prompt = SYSTEM_PROMPT_1
    # max_tokens = MAX_TOKENS
    # temperature = TEMPERATURE
    # frequence_penalty = FREQUENCE_PENALTY
    # presence_penalty = PRESENCE_PENALTY
    # ckpt_info = "None"

    print(f'Begin importing models')
    # from multipurpose_chatbot.globals import MODEL_ENGINE
    from multipurpose_chatbot.demos import get_demo_class
    
    # print(f'{MODEL_ENGINE=}')

    # demo_chat = create_chat_demo()
    # demo_file_upload = create_file_upload_demo()
    # demo_free_form = create_free_form_generation_demo()
    # demo_rag_chat = create_rag_chat_demo()

    # demo_chat = get_demo_class("VisionChatInterfaceDemo")().create_demo()
    demo_chat = get_demo_class("VisionDocChatInterfaceDemo")().create_demo()
    # demo_chat = get_demo_class("ChatInterfaceDemo")().create_demo()
    # demo_chat = get_demo_class("RagVisionChatInterfaceDemo")().create_demo()
    # demo_rag_chat = get_demo_class("RagChatInterfaceDemo")().create_demo()

    descriptions = model_desc
    descriptions += f"<br> {path_markdown.format(model_path=model_path)}"

    demo = CustomTabbedInterface(
        interface_list=[
            demo_chat, 
            # demo_rag_chat,
            # demo_free_form,
            # demo_file_upload, 
        ],
        tab_names=[
            "Chat Interface", 
            # "RAG Chat Interface", 
            # "Completion",
            # "Batch Inference", 
        ],
        title=f"{MODEL_TITLE}",
        description=descriptions,
    )

    demo.title = MODEL_NAME
    
    with demo:
        gr.Markdown(cite_markdown)
        

    demo.queue(api_open=False)
    return demo




if __name__ == "__main__":
    demo = launch_demo()
    if PROXY is not None and PROXY != "":
        demo.launch(server_port=PORT, root_path=PROXY, show_api=False, allowed_paths=allowed_paths)
    else:
        demo.launch(server_port=PORT, show_api=False, allowed_paths=allowed_paths)