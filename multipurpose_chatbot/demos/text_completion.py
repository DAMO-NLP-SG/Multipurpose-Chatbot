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
)

from ..globals import MODEL_ENGINE


def generate_text_completion_stream_engine(
    message: str, 
    temperature: float, 
    max_tokens: int, 
    stop_strings: str = '<s>,</s>,<|im_start|>,<|im_end|>',
):
    global MODEL_ENGINE
    temperature = float(temperature)
    # ! remove frequency_penalty
    # frequency_penalty = float(frequency_penalty)
    max_tokens = int(max_tokens)
    # message = message.strip()
    stop_strings = [x.strip() for x in stop_strings.strip().split(",")]
    stop_strings = list(set(stop_strings + ['</s>', '<|im_start|>', '<|im_end|>']))
    if message.strip() != message:
        gr.Warning(f'There are preceding/trailing spaces in the message, may lead to unexpected behavior')
    if len(message) == 0:
        raise gr.Error("The message cannot be empty!")
    num_tokens = len(MODEL_ENGINE.tokenizer.encode(message))
    if num_tokens >= MODEL_ENGINE.max_position_embeddings - 128:
        raise gr.Error(f"Conversation or prompt is too long ({num_tokens} toks), please clear the chatbox or try shorter input.")
    
    outputs = None
    response = None
    num_tokens = -1
    for j, outputs in enumerate(MODEL_ENGINE.generate_yield_string(
        prompt=message,
        temperature=temperature,
        max_tokens=max_tokens,
        stop_strings=stop_strings,
    )):
        if isinstance(outputs, tuple):
            response, num_tokens = outputs
        else:
            response, num_tokens = outputs, -1
        yield message + response, f"{num_tokens} tokens"
    
    if response is not None:
        yield message + response, f"{num_tokens} tokens"

    
@register_demo
class TextCompletionDemo(BaseDemo):
    @property
    def tab_name(self):
        return "Text Completion"
    
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
        # frequence_penalty = FREQUENCE_PENALTY
        # presence_penalty = PRESENCE_PENALTY
        max_tokens = max_tokens // 2

        description = description or f"""Put any context string (like few-shot prompts)"""

        with gr.Blocks() as demo_text_completion:
            if title:
                gr.Markdown(title)
            if description:
                gr.Markdown(description)
            with gr.Row():
                txt = gr.Textbox(
                    scale=4,
                    lines=16,
                    show_label=False,
                    placeholder="Enter any free form text and submit",
                    container=False,
                )
            with gr.Row():
                submit_button = gr.Button('Submit', variant='primary', scale=9)
                stop_button = gr.Button('Stop', variant='stop', scale=9, visible=False)
                num_tokens = Textbox(
                    container=False,
                    show_label=False,
                    label="num_tokens",
                    placeholder="0 tokens",
                    scale=1,
                    interactive=False,
                    min_width=10
                )
            with gr.Row():
                temp_input = gr.Number(value=temperature, label='Temperature', info="Higher -> more random")
                length_input = gr.Number(value=max_tokens, label='Max tokens', info='Increase if want more generation')
                stop_strings = gr.Textbox(value="<s>,</s>,<|im_start|>,<|im_end|>", label='Stop strings', info='Comma-separated string to stop generation only in FEW-SHOT mode', lines=1)
            examples = gr.Examples(
                examples=[
                    ["The following is the recite the declaration of independence:",]
                ],
                inputs=[txt, temp_input, length_input, stop_strings],
                # outputs=[txt]
            )
            # ! Handle stop button
            submit_trigger = submit_button.click
            submit_event = submit_button.click(
                # submit_trigger,
                generate_text_completion_stream_engine, 
                [txt, temp_input, length_input, stop_strings], 
                [txt, num_tokens],
                # api_name=False,
                # queue=False,
            )
            
            submit_trigger(
                lambda: (
                    Button(visible=False), Button(visible=True),
                ),
                None,
                [submit_button, stop_button],
                api_name=False,
                queue=False,
            )
            submit_event.then(
                lambda: (Button(visible=True), Button(visible=False)),
                None,
                [submit_button, stop_button],
                api_name=False,
                queue=False,
            )
            stop_button.click(
                None,
                None,
                None,
                cancels=submit_event,
                api_name=False,
            )
            
        return demo_text_completion