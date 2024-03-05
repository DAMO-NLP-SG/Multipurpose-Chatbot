import os
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
import time
from mlx_lm import load, generate
from mlx_lm.utils import generate_step

from .base_engine import BaseEngine

"""
generate(model: mlx.nn.layers.base.Module, tokenizer: transformers.tokenization_utils.PreTrainedTokenizer, prompt: str, temp: float = 0.0, max_tokens: int = 100, verbose: bool = False, formatter: Callable = None, repetition_penalty: Optional[float] = None, repetition_context_size: Optional[int] = None) -> str
    Generate text from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       temp (float): The temperature for sampling (default 0).
       max_tokens (int): The maximum number of tokens (default 100).
       verbose (bool): If ``True``, print tokens and timing information
           (default ``False``).
       formatter (Optional[Callable]): A function which takes a token and a
           probability and displays it.
       repetition_penalty (float, optional): The penalty factor for repeating tokens.
       repetition_context_size (int, optional): The number of tokens to consider for repetition penalty.


"""

MODEL_PATH = os.environ.get("MODEL_PATH", "./seal-13b-chat-a")

def generate_string(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    temp: float = 0.0,
    max_tokens: int = 100,
    verbose: bool = False,
    formatter: Callable = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = None,
    stop_strings: Optional[Tuple[str]] = None
):
    prompt_tokens = mx.array(tokenizer.encode(prompt))
    stop_strings = stop_strings if stop_strings is None or isinstance(stop_strings, tuple) else tuple(stop_strings)
    assert stop_strings is None or isinstance(stop_strings, tuple), f'invalid {stop_strings}'

    tic = time.perf_counter()
    tokens = []
    skip = 0
    REPLACEMENT_CHAR = "\ufffd"

    for (token, prob), n in zip(
        generate_step(
            prompt_tokens,
            model,
            temp,
            repetition_penalty,
            repetition_context_size,
        ),
        range(max_tokens),
    ):
        if token == tokenizer.eos_token_id:
            break
        if n == 0:
            prompt_time = time.perf_counter() - tic
            tic = time.perf_counter()
        tokens.append(token.item())
        if stop_strings is not None:
            token_string = tokenizer.decode(tokens).replace(REPLACEMENT_CHAR, "")
            if token_string.strip().endswith(stop_strings):
                break
    token_string = tokenizer.decode(tokens).replace(REPLACEMENT_CHAR, "")
    return token_string
    


def generate_yield_string(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    temp: float = 0.0,
    max_tokens: int = 100,
    verbose: bool = False,
    formatter: Callable = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = None,
    stop_strings: Optional[Tuple[str]] = None
):
    """
    Generate text from the model.
    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       temp (float): The temperature for sampling (default 0).
       max_tokens (int): The maximum number of tokens (default 100).
       verbose (bool): If ``True``, print tokens and timing information
           (default ``False``).
       formatter (Optional[Callable]): A function which takes a token and a
           probability and displays it.
       repetition_penalty (float, optional): The penalty factor for repeating tokens.
       repetition_context_size (int, optional): The number of tokens to consider for repetition penalty.
    """
    if verbose:
        print("=" * 10)
        print("Prompt:", prompt)
    stop_strings = stop_strings if stop_strings is None or isinstance(stop_strings, tuple) else tuple(stop_strings)
    assert stop_strings is None or isinstance(stop_strings, tuple), f'invalid {stop_strings}'
    prompt_tokens = mx.array(tokenizer.encode(prompt))
    tic = time.perf_counter()
    tokens = []
    skip = 0
    REPLACEMENT_CHAR = "\ufffd"
    for (token, prob), n in zip(
        generate_step(
            prompt_tokens,
            model,
            temp,
            repetition_penalty,
            repetition_context_size,
        ),
        range(max_tokens),
    ):
        if token == tokenizer.eos_token_id:
            break
        # if n == 0:
        #     prompt_time = time.perf_counter() - tic
        #     tic = time.perf_counter()
        tokens.append(token.item())
        # if verbose:
        #     s = tokenizer.decode(tokens)
        #     if formatter:
        #         formatter(s[skip:], prob.item())
        #         skip = len(s)
        #     elif REPLACEMENT_CHAR not in s:
        #         print(s[skip:], end="", flush=True)
        #         skip = len(s)
        token_string = tokenizer.decode(tokens).replace(REPLACEMENT_CHAR, "")
        yield token_string
        if stop_strings is not None and token_string.strip().endswith(stop_strings):
            break

    # token_count = len(tokens)
    # token_string = tokenizer.decode(tokens).replace(REPLACEMENT_CHAR, "")

    # if verbose:
    #     print(token_string[skip:], flush=True)
    #     gen_time = time.perf_counter() - tic
    #     print("=" * 10)
    #     if token_count == 0:
    #         print("No tokens generated for this prompt")
    #         return
    #     prompt_tps = prompt_tokens.size / prompt_time
    #     gen_tps = (token_count - 1) / gen_time
    #     print(f"Prompt: {prompt_tps:.3f} tokens-per-sec")
    #     print(f"Generation: {gen_tps:.3f} tokens-per-sec")

    # return token_string


class MlxEngine(BaseEngine):
    # (native) nguyenxuanphi@B-63TF46X3-2249 seallms % python -m mlx_lm.generate --model seallm-v2-199680-seallm-chatml-sts-mlx --max-tokens 1024 --prompt "<|im_start|>user\nGiải thích thuyết tương đối hẹp.<|im_end|><|im_start|>assistant\n"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._model = None
        self._tokenizer = None

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    def load_model(self, ):
        # self.model, self._tokenizer = load("seallm-v2-199680-seallm-chatml-sts-mlx")
        model_path = "/Users/nguyenxuanphi/Desktop/projects/cache/seallms/seallm-v2-199680-seallm-chatml-sts-mlx"
        self._model, self._tokenizer = load(model_path)
        print(f'Load MLX model from {model_path}')
    

    def generate_yield_string(self, prompt, temperature, max_tokens, stop_strings: Optional[Tuple[str]] = None, **kwargs):
        num_tokens = len(self.tokenizer.encode(prompt))
        response = None
        for response in generate_yield_string(
            self._model, self._tokenizer,
            prompt, temp=temperature, max_tokens=max_tokens,
            repetition_penalty=kwargs.get("repetition_penalty", None),
            stop_strings=stop_strings,
        ):
            yield response, num_tokens
        if response is not None:
            full_text = prompt + response
            num_tokens = len(self.tokenizer.encode(full_text))
            yield response, num_tokens

    def batch_generate(self, prompts, temperature, max_tokens, stop_strings: Optional[Tuple[str]] = None, **kwargs):
        """
        ! MLX does not support 
        """
        responses = [
            generate_string(
                self._model, self._tokenizer,
                s, temp=temperature, max_tokens=max_tokens,
                repetition_penalty=kwargs.get("repetition_penalty", None),
                stop_strings=stop_strings,
            )
            for s in prompts
        ]
        return responses

    
    


    


