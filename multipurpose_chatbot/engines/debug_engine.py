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
FAKE_MODEL_PATH = "/Users/nguyenxuanphi/Desktop/projects/cache/seallms/seallm-v2-199680-seallm-chatml-sts-mlx"



class DebugEngine(BaseEngine):
    """Will repeat the prompt only"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._model = None
        self._tokenizer = None

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(FAKE_MODEL_PATH, trust_remote_code=True)
        return self._tokenizer

    def load_model(self):
        print("Load fake model")
    
    def generate_yield_string(self, prompt, temperature, max_tokens, stop_strings: Optional[Tuple[str]] = None, **kwargs):
        num_tokens = len(self.tokenizer.encode(prompt))
        response = "Wow that's very very cool"
        for i in range(len(response)):
            time.sleep(0.01)
            yield response[:i], num_tokens

        num_tokens = len(self.tokenizer.encode(prompt + response))
        yield response, num_tokens
    
    def batch_generate(self, prompts, temperature, max_tokens, stop_strings: Optional[Tuple[str]] = None, **kwargs):
        return [p + " -- Test" for p in prompts]
