try:
    import spaces
    def maybe_spaces_gpu(fn):
        fn = spaces.GPU(fn)
        return fn
except ModuleNotFoundError:
    print(f'Cannot import hf `spaces` with `import spaces`.')
    def maybe_spaces_gpu(fn):
        return fn
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

from .chat_interface import (
    CHAT_EXAMPLES,
    DATETIME_FORMAT,
    gradio_history_to_conversation_prompt,
    gradio_history_to_openai_conversations,
    get_datetime_string,
    format_conversation,
    chat_response_stream_multiturn_engine,
    CustomizedChatInterface,
    ChatInterfaceDemo
)

from .langchain_web_search import (
    AnyEnginePipeline,
    ChatAnyEnginePipeline,
    create_web_search_engine,
)


web_search_llm = None
web_search_chat_model = None
web_search_engine = None
web_search_agent = None


@maybe_spaces_gpu
def chat_web_search_response_stream_multiturn_engine(
    message: str, 
    history: List[Tuple[str, str]], 
    temperature: float, 
    max_tokens: int, 
    system_prompt: Optional[str] = SYSTEM_PROMPT,
):
    # global web_search_engine, web_search_llm, web_search_chat_model, web_search_agent, MODEL_ENGINE
    global web_search_llm, web_search_chat_model, agent_executor, MODEL_ENGINE
    temperature = float(temperature)
    # ! remove frequency_penalty
    # frequency_penalty = float(frequency_penalty)
    max_tokens = int(max_tokens)
    message = message.strip()
    if len(message) == 0:
        raise gr.Error("The message cannot be empty!")
    
    response_output = agent_executor.invoke({"input": message})
    print(response_output)
    response = response_output['output']
    
    full_prompt = gradio_history_to_conversation_prompt(message.strip(), history=history, system_prompt=system_prompt)
    num_tokens = len(MODEL_ENGINE.tokenizer.encode(full_prompt))
    yield response, num_tokens

    # # ! skip safety
    # if DATETIME_FORMAT in system_prompt:
    #     # ! This sometime works sometimes dont
    #     system_prompt = system_prompt.format(cur_datetime=get_datetime_string())
    # full_prompt = gradio_history_to_conversation_prompt(message.strip(), history=history, system_prompt=system_prompt)
    # # ! length checked
    # num_tokens = len(MODEL_ENGINE.tokenizer.encode(full_prompt))
    # if num_tokens >= MODEL_ENGINE.max_position_embeddings - 128:
    #     raise gr.Error(f"Conversation or prompt is too long ({num_tokens} toks), please clear the chatbox or try shorter input.")
    # print(full_prompt)
    # outputs = None
    # response = None
    # num_tokens = -1
    # for j, outputs in enumerate(MODEL_ENGINE.generate_yield_string(
    #     prompt=full_prompt,
    #     temperature=temperature,
    #     max_tokens=max_tokens,
    # )):
    #     if isinstance(outputs, tuple):
    #         response, num_tokens = outputs
    #     else:
    #         response, num_tokens = outputs, -1
    #     yield response, num_tokens
        
    # print(format_conversation(history + [[message, response]]))

    # if response is not None:
    #     yield response, num_tokens





@register_demo
class WebSearchChatInterfaceDemo(BaseDemo):
    @property
    def tab_name(self):
        return "Web Search"
    
    def create_demo(
            self, 
            title: str | None = None, 
            description: str | None = None, 
            **kwargs
        ) -> gr.Blocks:
        global web_search_llm, web_search_chat_model, agent_executor
        system_prompt = kwargs.get("system_prompt", SYSTEM_PROMPT)
        max_tokens = kwargs.get("max_tokens", MAX_TOKENS)
        temperature = kwargs.get("temperature", TEMPERATURE)
        model_name = kwargs.get("model_name", MODEL_NAME)
        # frequence_penalty = FREQUENCE_PENALTY
        # presence_penalty = PRESENCE_PENALTY
        # create_web_search_engine()
        description = description or "At the moment, Web search is only **SINGLE TURN**, only works well in **English** and may respond unnaturally!"

        web_search_llm, web_search_chat_model, agent_executor = create_web_search_engine()

        demo_chat = CustomizedChatInterface(
            chat_web_search_response_stream_multiturn_engine,
            chatbot=gr.Chatbot(
                label=model_name,
                bubble_full_width=False,
                latex_delimiters=[
                    { "left": "$", "right": "$", "display": False},
                    { "left": "$$", "right": "$$", "display": True},
                ],
                show_copy_button=True,
                layout="panel" if USE_PANEL else "bubble",
                height=CHATBOT_HEIGHT,
            ),
            textbox=gr.Textbox(placeholder='Type message', lines=1, max_lines=128, min_width=200, scale=8),
            submit_btn=gr.Button(value='Submit', variant="primary", scale=0),
            title=title,
            description=description,
            additional_inputs=[
                gr.Number(value=temperature, label='Temperature (higher -> more random)'), 
                gr.Number(value=max_tokens, label='Max generated tokens (increase if want more generation)'), 
                # gr.Number(value=frequence_penalty, label='Frequency penalty (> 0 encourage new tokens over repeated tokens)'), 
                # gr.Number(value=presence_penalty, label='Presence penalty (> 0 encourage new tokens, < 0 encourage existing tokens)'), 
                gr.Textbox(value=system_prompt, label='System prompt', lines=4, interactive=False)
            ], 
            examples=[
                # ["What is 234.56 raised to the 0.43 power?"],
                ["What is Langchain?"],
                ["Give me latest news about Lawrence Wong."],
                # ["Chủ tịch nước Việt Nam hiện này là ai? Hãy sử dụng trình tìm kiếm."],
                # ["Chủ tịch nước Việt Nam hiện này là ai?"],
                ['What did Jerome Powell say today?'],
                ['What is the best model on the LMSys leaderboard?'],
                ['Where does Messi play right now?'],
            ],
            # ] + CHAT_EXAMPLES,
            cache_examples=False
        )
        return demo_chat

