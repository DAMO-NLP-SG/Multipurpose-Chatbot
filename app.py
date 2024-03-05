# Copyright: DAMO Academy, Alibaba Group
# By Xuan Phi Nguyen at DAMO Academy, Alibaba Group

# Description:
"""
Demo script to launch Language chat model 
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
If you find our project useful, hope you can star our repo and cite our repo as follows:
```
@article{multipurpose_chatbot_2024,
  author = {Xuan-Phi Nguyen, },
  title = {Multipurpose Chatbot},
  year = 2024,
}
```
"""


demo_and_tab_names = {
    "VisionDocChatInterfaceDemo": "Vision Doc Chat",
    "ChatInterfaceDemo": "Chat",
    # "RagChatInterfaceDemo": "RAG Chat",
}


set_documentation_group("component")


def launch_demo():
    global demo, MODEL_ENGINE
    model_desc = MODEL_DESC
    model_path = MODEL_PATH

    print(f'Begin importing models')
    from multipurpose_chatbot.demos import get_demo_class

    demos = {
        k: get_demo_class(k)().create_demo()
        for k in demo_and_tab_names.keys()
    }
    # demo_chat = get_demo_class("VisionChatInterfaceDemo")().create_demo()
    # demo_chat = get_demo_class("VisionDocChatInterfaceDemo")().create_demo()
    # demo_chat = get_demo_class("ChatInterfaceDemo")().create_demo()
    # demo_chat = get_demo_class("RagVisionChatInterfaceDemo")().create_demo()
    # demo_rag_chat = get_demo_class("RagChatInterfaceDemo")().create_demo()

    descriptions = model_desc
    descriptions += f"<br> {path_markdown.format(model_path=model_path)}"

    demo = CustomTabbedInterface(
        interface_list=list(demos.values()),
        tab_names=list(demo_and_tab_names.values()),
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