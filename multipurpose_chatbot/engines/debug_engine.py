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

from ..configs import (
    MODEL_PATH,
)

FAKE_MODEL_PATH = os.environ.get("FAKE_MODEL_PATH", MODEL_PATH)
FAKE_RESPONSE = "Wow that's very very cool, please try again."


class DebugEngine(BaseEngine):
    """
    It will always yield FAKE_RESPONSE
    """

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
        print(f"Load fake model with tokenizer: {self.tokenizer}")
    
    def generate_yield_string(self, prompt, temperature, max_tokens, stop_strings: Optional[Tuple[str]] = None, **kwargs):

        num_tokens = len(self.tokenizer.encode(prompt))
        response = FAKE_RESPONSE
        for i in range(len(response)):
            time.sleep(0.01)
            yield response[:i], num_tokens

        num_tokens = len(self.tokenizer.encode(prompt + response))
        yield response, num_tokens
    
    def batch_generate(self, prompts, temperature, max_tokens, stop_strings: Optional[Tuple[str]] = None, **kwargs):
        return [p + " -- Test" for p in prompts]
