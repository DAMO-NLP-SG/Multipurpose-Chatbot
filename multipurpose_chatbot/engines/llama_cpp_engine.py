import os
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
from typing import AsyncGenerator, Callable, Literal, Union, cast

from gradio_client.documentation import document, set_documentation_group

from typing import List, Optional, Union, Dict, Tuple
from tqdm.auto import tqdm
from huggingface_hub import snapshot_download
import types

from gradio.components import Button
from gradio.events import Dependency, EventListenerMethod

import types
import sys

from .base_engine import BaseEngine

# ! Remember to use static cache

from ..configs import (
    MODEL_PATH,
    DEFAULT_CHAT_TEMPLATE,
    N_CTX,
    N_GPU_LAYERS,
)



def encode_tokenize(self, prompt: str, **kwargs):
    """Mimic behavior of transformers tokenizer"""
    prompt_tokens: List[int] = (
        (
            self.tokenize(prompt.encode("utf-8"), special=True)
            if prompt != ""
            else [self.token_bos()]
        )
        if isinstance(prompt, str)
        else prompt
    )
    return prompt_tokens


conversations = [
    {"role": "system", "content": "You are good."},
    {"role": "user", "content": "Hello."},
    {"role": "assistant", "content": "Hi."},
]


class LlamaCppEngine(BaseEngine):
    """
    need to create an engine.tokenizer.encode(text) method
    """
    @property
    def max_position_embeddings(self) -> int:
        # raise ValueError
        return self._model.context_params.n_ctx
    
    def apply_chat_template(self, conversations, add_generation_prompt: bool, add_special_tokens=False, **kwargs) -> str:
        """
        return string convo, add_special_tokens should be added later
        remember to remove <s> if any,
        """
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        formatter = Jinja2ChatFormatter(
            template=self._model.metadata['tokenizer.chat_template'],
            # bos_token=self._model._model.token_get_text(self._model.token_bos()),
            bos_token="",
            eos_token=self._model._model.token_get_text(self._model.token_eos()),
            add_generation_prompt=add_generation_prompt,
        )

        full_prompt = formatter(messages=conversations).prompt
        # ! it may has bos
        return full_prompt

    @property
    def tokenizer(self):
        return self._model
    
    def load_model(self):
        # from transformers import AutoTokenizer, AutoModelForCausalLM
        
        from llama_cpp import Llama
        self.model_path = MODEL_PATH
        self._model = Llama(
            model_path=self.model_path,
            n_gpu_layers=N_GPU_LAYERS, # Uncomment to use GPU acceleration
            # seed=1337, # Uncomment to set a specific seed
            n_ctx=N_CTX, # Uncomment to increase the context window
        )
        self._tokenizer = self._model
        self._model.encode = types.MethodType(encode_tokenize, self._model)
        print(f'Load model: {self.model_path=} | {N_GPU_LAYERS=} | {N_CTX=}')

    def generate_yield_string(self, prompt, temperature, max_tokens, stop_strings: Optional[Tuple[str]] = None, **kwargs):
        stop_strings = list(stop_strings) if stop_strings is not None else []
        stop_strings = list(set(stop_strings + ["</s>", "<|im_end|>"]))
        generator = self._model(
            prompt,
            max_tokens=max_tokens, # Generate up to 32 tokens, set to None to generate up to the end of the context window
            temperature=temperature,
            stop=stop_strings, # Stop generating just before the model would generate a new question
            stream=True,
        )
        response = ""
        num_tokens = len(self.tokenizer.encode(prompt))
        for g in generator:
            response += g['choices'][0]['text']
            yield response, num_tokens

        if response is not None and len(response) > 0:
            num_tokens = len(self.tokenizer.encode(prompt + response))
            yield response, num_tokens


