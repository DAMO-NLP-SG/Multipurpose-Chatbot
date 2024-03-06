import os
import numpy as np
from huggingface_hub import snapshot_download
# ! Avoid importing transformers
# from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
import time


class BaseEngine(object):
    def __init__(self, **kwargs) -> None:
        pass

    @property
    def max_position_embeddings(self) -> int:
        return 10000

    @property
    def tokenizer(self):
        raise NotImplementedError
    
    def load_model(self, ):
        raise NotImplementedError

    def apply_chat_template(self, conversations, add_generation_prompt: bool, add_special_tokens=False, **kwargs) -> str:
        """
        return string convo, add_special_tokens should be added later
        """
        bos_token = self.tokenizer.bos_token
        eos_token = self.tokenizer.eos_token
        if not add_special_tokens:
            # prevent bos being added to string
            self.tokenizer.bos_token = ""
            self.tokenizer.eos_token = ""
        full_prompt = self.tokenizer.apply_chat_template(
            conversations, add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
        self.tokenizer.bos_token = bos_token
        self.tokenizer.eos_token = eos_token
        return full_prompt
    
