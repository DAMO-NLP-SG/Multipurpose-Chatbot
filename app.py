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

from multipurpose_chatbot.configs import (
    MODEL_TITLE,
    MODEL_DESC,
    MODEL_INFO,
    CITE_MARKDOWN,
    ALLOWED_PATHS,
    PROXY,
    PORT,
    MODEL_PATH,
    MODEL_NAME,
    BACKEND,
    DEMOS,
)


demo = None





def launch_demo():
    global demo, MODEL_ENGINE
    model_desc = MODEL_DESC
    model_path = MODEL_PATH

    print(f'Begin importing models')
    from multipurpose_chatbot.demos import get_demo_class

    # demos = {
    #     k: get_demo_class(k)().create_demo()
    #     for k in demo_and_tab_names.keys()
    # }
    print(f'{DEMOS=}')
    demo_class_objects = {
        k: get_demo_class(k)()
        for k in DEMOS
    }
    demos = {
        k: get_demo_class(k)().create_demo()
        for k in DEMOS
    }
    demos_names = [x.tab_name for x in demo_class_objects.values()]

    descriptions = model_desc
    if MODEL_INFO is not None and MODEL_INFO != "":
        descriptions += (
            f"<br>" + 
            MODEL_INFO.format(model_path=model_path)
        )

    demo = CustomTabbedInterface(
        interface_list=list(demos.values()),
        tab_names=demos_names,
        title=f"{MODEL_TITLE}",
        description=descriptions,
    )

    demo.title = MODEL_NAME
    
    with demo:
        gr.Markdown(CITE_MARKDOWN)
        
    demo.queue(api_open=False)
    return demo



if __name__ == "__main__":
    demo = launch_demo()
    if PROXY is not None and PROXY != "":
        print(f'{PROXY=} {PORT=}')
        demo.launch(server_port=PORT, root_path=PROXY, show_api=False, allowed_paths=ALLOWED_PATHS)
    else:
        demo.launch(server_port=PORT, show_api=False, allowed_paths=ALLOWED_PATHS)


