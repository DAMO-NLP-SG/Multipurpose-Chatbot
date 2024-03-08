import os
from gradio.themes import ThemeClass as Theme
import numpy as np
import argparse
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
from typing import AsyncGenerator, Callable, Literal, Union, cast, Generator

from gradio_client.documentation import document, set_documentation_group
from gradio.components import Button, Component
from gradio.events import Dependency, EventListenerMethod
from typing import List, Optional, Union, Dict, Tuple
from tqdm.auto import tqdm
from huggingface_hub import snapshot_download


import inspect
from typing import AsyncGenerator, Callable, Literal, Union, cast

import anyio
from gradio_client import utils as client_utils
from gradio_client.documentation import document

from gradio.blocks import Blocks
from gradio.components import (
    Button,
    Chatbot,
    Component,
    Markdown,
    State,
    Textbox,
    get_component_instance,
)
from gradio.events import Dependency, on
from gradio.helpers import create_examples as Examples  # noqa: N812
from gradio.helpers import special_args
from gradio.layouts import Accordion, Group, Row
from gradio.routes import Request
from gradio.themes import ThemeClass as Theme
from gradio.utils import SyncToAsyncIterator, async_iteration


from .base_demo import register_demo, get_demo_class, BaseDemo
from ..configs import (
    SYSTEM_PROMPT,
    MODEL_NAME,
    MAX_TOKENS,
    TEMPERATURE,
    USE_PANEL,
    CHATBOT_HEIGHT,
)

from ..globals import MODEL_ENGINE

from .chat_interface import gradio_history_to_conversation_prompt

# Batch inference file upload
ENABLE_BATCH_INFER = bool(int(os.environ.get("ENABLE_BATCH_INFER", "1")))
BATCH_INFER_MAX_ITEMS = int(os.environ.get("BATCH_INFER_MAX_ITEMS", "100"))
BATCH_INFER_MAX_FILE_SIZE = int(os.environ.get("BATCH_INFER_MAX_FILE_SIZE", "500"))
BATCH_INFER_MAX_PROMPT_TOKENS = int(os.environ.get("BATCH_INFER_MAX_PROMPT_TOKENS", "4000"))
BATCH_INFER_SAVE_TMP_FILE = os.environ.get("BATCH_INFER_SAVE_TMP_FILE", "./tmp/pred.json")



FILE_UPLOAD_DESCRIPTION = f"""Upload JSON file as list of dict with < {BATCH_INFER_MAX_ITEMS} items, \
each item has `prompt` key. We put guardrails to enhance safety, so do not input any harmful content or personal information! Re-upload the file after every submit. See the examples below.
```
[ {{"id": 0, "prompt": "Hello world"}} ,  {{"id": 1, "prompt": "Hi there?"}}]
```
"""

def validate_file_item(filename, index, item: Dict[str, str]):
    """
    check safety for items in files
    """
    global MODEL_ENGINE
    message = item['prompt'].strip()

    if len(message) == 0:
        raise gr.Error(f'Prompt {index} empty')

    num_tokens = len(MODEL_ENGINE.tokenizer.encode(message))
    if num_tokens >= MODEL_ENGINE.max_position_embeddings - 128:
        raise gr.Error(f"Conversation or prompt is too long ({num_tokens} toks), please clear the chatbox or try shorter input.")
    

def read_validate_json_files(files: Union[str, List[str]]):
    files = files if isinstance(files, list) else [files]
    filenames = [f.name for f in files]
    all_items = []
    for fname in filenames:
        # check each files
        print(f'Reading {fname}')
        with open(fname, 'r', encoding='utf-8') as f:
            items = json.load(f)
        assert isinstance(items, list), f'Data {fname} not list'
        assert all(isinstance(x, dict) for x in items), f'item in input file not list'
        assert all("prompt" in x for x in items), f'key prompt should be in dict item of input file'

        for i, x in enumerate(items):
            validate_file_item(fname, i, x)

        all_items.extend(items)

    if len(all_items) > BATCH_INFER_MAX_ITEMS:
        raise gr.Error(f"Num samples {len(all_items)} > {BATCH_INFER_MAX_ITEMS} allowed.")
    
    return all_items, filenames


def remove_gradio_cache(exclude_names=None):
    """remove gradio cache to avoid flooding"""
    import shutil
    for root, dirs, files in os.walk('/tmp/gradio/'):
        for f in files:
            # if not any(f in ef for ef in except_files):
            if exclude_names is None or not any(ef in f for ef in exclude_names):
                print(f'Remove: {f}')
                os.unlink(os.path.join(root, f))


def free_form_prompt(prompt, history=None, system_prompt=None):
    return prompt




def batch_inference_engine(
        files: Union[str, List[str]], 
        prompt_mode: str,
        temperature: float, 
        max_tokens: int, 
        stop_strings: str = "<s>,</s>,<|im_start|>",
        system_prompt: Optional[str] = SYSTEM_PROMPT,
):
    global MODEL_ENGINE
    temperature = float(temperature)
    max_tokens = int(max_tokens)
    stop_strings = [x.strip() for x in stop_strings.strip().split(",")]

    all_items, filenames = read_validate_json_files(files)

    # remove all items in /tmp/gradio/
    remove_gradio_cache(exclude_names=['upload_chat.json', 'upload_few_shot.json'])

    if prompt_mode == 'chat':
        prompt_format_fn = gradio_history_to_conversation_prompt
    elif prompt_mode == 'few-shot':
        from functools import partial
        prompt_format_fn = free_form_prompt 
    else:
        raise gr.Error(f'Wrong mode {prompt_mode}')
    
    full_prompts = [
        prompt_format_fn(
            x['prompt'], [], system_prompt=system_prompt
        )
        for i, x in enumerate(all_items)
    ]
    print(f'{full_prompts[0]}\n')

    full_num_tokens = [
        len(MODEL_ENGINE.tokenizer.encode(p))
        for p in full_prompts
    ]
    if any(x >= MODEL_ENGINE.max_position_embeddings - 128 for x in full_num_tokens):
        raise gr.Error(f"Some prompt is too long!")

    # ! batch inference
    responses = MODEL_ENGINE.batch_generate(
        full_prompts,
        temperature=temperature, max_tokens=max_tokens,
        stop_strings=stop_strings,
    )

    if len(responses) != len(all_items):
        raise gr.Error(f'inconsistent lengths {len(responses)} != {len(all_items)}')

    for res, item in zip(responses, all_items):
        item['response'] = res

    save_path = BATCH_INFER_SAVE_TMP_FILE
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_items, f, indent=4, ensure_ascii=False)
    
    print_items = all_items[:2]
    print(json.dumps(print_items, indent=4, ensure_ascii=False))
    return save_path, print_items


class BatchInferenceDemo(BaseDemo):
    def tab_name(self):
        return "Batch Inference"
    

    def create_demo(
            self, 
            title: str | None = None, 
            description: str | None = None, 
            **kwargs
        ) -> gr.Blocks:
        system_prompt = kwargs.get("system_prompt", SYSTEM_PROMPT)
        max_tokens = kwargs.get("max_tokens", MAX_TOKENS)
        temperature = kwargs.get("temperature", TEMPERATURE)
        model_name = kwargs.get("model_name", MODEL_NAME)

        
        demo_file_upload = gr.Interface(
            batch_inference_engine,
            inputs=[
                gr.File(file_count='single', file_types=['json']),
                gr.Radio(["chat", "few-shot"], value='chat', label="Chat or Few-shot mode", info="Chat's output more user-friendly, Few-shot's output more consistent with few-shot patterns."),
                gr.Number(value=temperature, label='Temperature', info="Higher -> more random"), 
                gr.Number(value=max_tokens, label='Max tokens', info='Increase if want more generation'), 
                # gr.Number(value=frequence_penalty, label='Frequency penalty', info='> 0 encourage new tokens over repeated tokens'), 
                # gr.Number(value=presence_penalty, label='Presence penalty', info='> 0 encourage new tokens, < 0 encourage existing tokens'), 
                gr.Textbox(value="<s>,</s>,<|im_start|>", label='Stop strings', info='Comma-separated string to stop generation only in FEW-SHOT mode', lines=1),
                # gr.Number(value=0, label='current_time', visible=False), 
                gr.Textbox(value=system_prompt, label='System prompt', lines=4)
            ],
            outputs=[
                # "file",
                gr.File(label="Generated file"),
                # "json"
                gr.JSON(label='Example outputs (display 2 samples)')
            ],
            description=FILE_UPLOAD_DESCRIPTION,
            allow_flagging=False,
            examples=[
                ["upload_chat.json", "chat"],
                ["upload_few_shot.json", "few-shot"],
            ],
            cache_examples=False,
        )
        return demo_file_upload