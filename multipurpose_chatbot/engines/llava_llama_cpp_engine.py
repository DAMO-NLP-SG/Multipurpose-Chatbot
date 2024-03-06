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
    IMAGE_TOKEN,
    IMAGE_TOKEN_INTERACTIVE,
    IMAGE_TOKEN_LENGTH,
    MAX_PACHES,
)

from .llama_cpp_engine import (
    encode_tokenize,
    LlamaCppEngine,
)



# resource: https://llama-cpp-python.readthedocs.io/en/latest/#multi-modal-models

import base64

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"


# file_path = 'file_path.png'
# data_uri = image_to_base64_data_uri(file_path)

# data_uri = image_to_base64_data_uri(file_path)

# messages = [
#     {"role": "system", "content": "You are an assistant who perfectly describes images."},
#     {
#         "role": "user",
#         "content": [
#             {"type": "image_url", "image_url": {"url": data_uri }},
#             {"type" : "text", "text": "Describe this image in detail please."}
#         ]
#     }
# ]
    

def llava_15_chat_handler_call(
        self,
        *,
        llama: Any,
        # messages: List[Any],
        prompt: Union[str, List[int]],
        image_data_uris: Optional[List[Any]] = None,
        image_token: str = None,
        functions: Optional[List[Any]] = None,
        function_call: Optional[Any] = None,
        tools: Optional[List[Any]] = None,
        tool_choice: Optional[Any] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        response_format: Optional[
            Any
        ] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        logits_processor: Optional[Any] = None,
        grammar: Optional[Any] = None,
        **kwargs,  # type: ignore
):
    from llama_cpp.llama_chat_format import (
        ctypes,
        suppress_stdout_stderr,
    )
    assert (
        llama.context_params.logits_all is True
    )  # BUG: logits_all=True is required for llava
    assert self.clip_ctx is not None
    # ! split prompt into different parts
    assert image_token is not None
    prompt_parts = prompt.split(image_token)
    # assert len(prompt_parts)
    assert len(prompt_parts) == len(image_data_uris) + 1, f'invalid {len(prompt_parts)=} != {len(image_data_uris)=}'
    llama.reset()
    prefix = prompt_parts[0]
    remaining_texts = prompt_parts[1:]
    llama.reset()
    llama.eval(llama.tokenize(prefix.encode("utf8"), add_bos=True))
    for index, (image_uri, prompt_p) in enumerate(zip(image_data_uris, remaining_texts)):
        image_bytes = self.load_image(image_uri)
        import array
        data_array = array.array("B", image_bytes)
        c_ubyte_ptr = (
            ctypes.c_ubyte * len(data_array)
        ).from_buffer(data_array)
        with suppress_stdout_stderr(disable=self.verbose):
            embed = (
                self._llava_cpp.llava_image_embed_make_with_bytes(
                    self.clip_ctx,
                    llama.context_params.n_threads,
                    c_ubyte_ptr,
                    len(image_bytes),
                )
            )
        try:
            n_past = ctypes.c_int(llama.n_tokens)
            n_past_p = ctypes.pointer(n_past)
            with suppress_stdout_stderr(disable=self.verbose):
                self._llava_cpp.llava_eval_image_embed(
                    llama.ctx,
                    embed,
                    llama.n_batch,
                    n_past_p,
                )
            assert llama.n_ctx() >= n_past.value
            llama.n_tokens = n_past.value
        finally:
            with suppress_stdout_stderr(disable=self.verbose):
                self._llava_cpp.llava_image_embed_free(embed)

        llama.eval(llama.tokenize(prompt_p.encode("utf8"), add_bos=False))
    assert llama.n_ctx() >= llama.n_tokens

    prompt = llama.input_ids[: llama.n_tokens].tolist()
    # from llava-1.5
    return llama.create_completion(
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        typical_p=typical_p,
        stream=stream,
        stop=stop,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repeat_penalty=repeat_penalty,
        tfs_z=tfs_z,
        mirostat_mode=mirostat_mode,
        mirostat_tau=mirostat_tau,
        mirostat_eta=mirostat_eta,
        model=model,
        logits_processor=logits_processor,
        grammar=grammar,
    )



class LlavaLlamaCppEngine(LlamaCppEngine):
    """
    Still in development, expect BUGS

    ERROR: could not know why
    objc[61055]: Class GGMLMetalClass is implemented in both miniconda3/envs/native/lib/python3.12/site-packages/llama_cpp/libllama.dylib (0x12cb40290) and miniconda3/envs/native/lib/python3.12/site-packages/llama_cpp/libllava.dylib (0x12d9c8290). One of the two will be used. Which one is undefined.

    """
    @property
    def image_token(self):
        return IMAGE_TOKEN
    
    def get_multimodal_tokens(self, full_prompt, image_paths=None):
        num_tokens = len(self.tokenizer.encode(full_prompt))
        for image_path in image_paths:
            num_tokens += IMAGE_TOKEN_LENGTH * MAX_PACHES
        return num_tokens
    
    def load_model(self):
        # from transformers import AutoTokenizer, AutoModelForCausalLM
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        model_dir = os.path.dirname(MODEL_PATH)
        self.chat_handler = Llava15ChatHandler(clip_model_path=os.path.join(model_dir, "mmproj.bin"))

        self.chat_handler.__call__ = types.MethodType(llava_15_chat_handler_call, self.chat_handler)
        
        self.model_path = MODEL_PATH
        self._model = Llama(
            model_path=self.model_path,
            n_gpu_layers=N_GPU_LAYERS, # Uncomment to use GPU acceleration
            # seed=1337, # Uncomment to set a specific seed
            chat_handler=self.chat_handler,
            n_ctx=N_CTX, # Uncomment to increase the context window
            logits_all=True, # needed to make llava work
        )
        self._tokenizer = self._model
        self._model.encode = types.MethodType(encode_tokenize, self._model)
        print(f'Load model: {self.model_path=} | {N_GPU_LAYERS=} | {N_CTX=}')
    
    def generate_yield_string(self, prompt, temperature, max_tokens, stop_strings: Optional[Tuple[str]] = None, **kwargs):
        image_paths = kwargs.get("image_paths", [])

        image_data_uris = [
            image_to_base64_data_uri(ip)
            for ip in image_paths
        ]
        
        stop_strings = list(stop_strings) if stop_strings is not None else []
        stop_strings = list(set(stop_strings + ["</s>", "<|im_end|>"]))
        # generator = self._model(
        generator = self.chat_handler(
            prompt=prompt,
            image_data_uris=image_data_uris,
            image_token=self.image_token,
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
    

"""

export MODEL_PATH
BACKEND=llama_cpp
MODEL_PATH=/Users/nguyenxuanphi/Desktop/projects/cache/seallms/SeaLLMs/SeaLLM-7B-v2-gguf/seallm-v2.chatml.Q4_K_M.gguf
N_CTX=4096
python app.py


export BACKEND=llava_llama_cpp
export MODEL_PATH=/Users/nguyenxuanphi/Desktop/projects/cache/llava/llava-1.5/ggml-model-q4_k.gguf
export N_CTX=4096
export IMAGE_TOKEN="<image>"
python app.py


"""