
import os
import numpy as np
import argparse
import torch
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

from gradio.components import Button
from gradio.events import Dependency, EventListenerMethod
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer
import types
import sys
from .base_engine import BaseEngine
from .transformers_engine import TransformersEngine, NewGenerationMixin

from ..configs import (
    STREAM_CHECK_MULTIPLE,
    STREAM_YIELD_MULTIPLE,
)

from ..configs import (
    STREAM_CHECK_MULTIPLE,
    STREAM_YIELD_MULTIPLE,
    IMAGE_TOKEN,
    IMAGE_TOKEN_INTERACTIVE,
    IMAGE_TOKEN_LENGTH,
    MAX_PACHES,
    DTYPE,
    DEVICE,
)

CODE_PATH = os.environ.get("CODE_PATH", "")
MODEL_PATH = os.environ.get("MODEL_PATH", "")

# IMAGE_TOKEN = "<image"

# IMAGE_LENGTH = 576
# MAX_PACHES = 1


# ! Still working on it....
# Should only do with 

"""
This is a chat between an inquisitive human and an AI assistant. Assume the role of the AI assistant. Read all the images carefully, and respond to the human's questions with informative, helpful, detailed and polite answers. 这是一个好奇的人类和一个人工智能助手之间的对话。假设你扮演这个AI助手的角色。仔细阅读所有的图像，并对人类的问题做出信息丰富、有帮助、详细的和礼貌的回答。

### Human: <image_placeholder>
Describe the cats and what they are doing in detail.
### Assistant:
"""

# prompt = "USER: <image>\nWhat are these?\nASSISTANT:"
# image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"

# conv_llava_llama_2 = Conversation(
#     system="You are a helpful language and vision assistant. "
#            "You are able to understand the visual content that the user provides, "
#            "and assist the user with a variety of tasks using natural language.",
#     roles=("USER", "ASSISTANT"),
#     version="llama_v2",
#     messages=(),
#     offset=0,
#     sep_style=SeparatorStyle.LLAMA_2,
#     sep="<s>",
#     sep2="</s>",
# )


LLAVA_CHAT_TEMPLATE = """"""


#   "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{{ bos_token }}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '</s>'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


if IMAGE_TOKEN != "<image>":
    print(f'WARNING!!!! {IMAGE_TOKEN=} is not <image>, this can lead to problems')


class Llava15TransformersEngine(TransformersEngine):
    """
    Llava 1.5 hardcoded
    """
    @property
    def image_token(self):
        return IMAGE_TOKEN

    @property
    def max_position_embeddings(self) -> int:
        return self._model.config.text_config.max_position_embeddings

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def processor(self):
        return self._processor
    

    def apply_chat_template(self, conversations, add_generation_prompt: bool, add_special_tokens=False, **kwargs) -> str:
        """
        return string convo, add_special_tokens should be added later
        """
        prompt = ""
        for turn in conversations:
            if turn['role'] == 'system':
                prompt += turn['content'] + "\n\n"
            elif turn['role'] == 'user':
                prompt += f"USER: {turn['content']}\n"
            elif turn['role'] == 'assistant':
                prompt += f"ASSISTANT: {turn['content']}\n"
        if add_generation_prompt:
            prompt += f"ASSISTANT:"
        return prompt
    

    def load_model(self):
        import requests
        from PIL import Image
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.model_path = model_path = MODEL_PATH
        self.torch_dtype = torch.bfloat16 if DTYPE == 'bfloat16' else torch.float16
        self.device_map = DEVICE
        print(f'Loading model from {model_path} on {self.device_map} with {self.torch_dtype} | LlavaForConditionalGeneration')

        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_PATH, 
            torch_dtype=self.torch_dtype, device_map=self.device_map, trust_remote_code=True
        ).eval()
        self._model.sample_old = self._model.sample
        # self._model.sample = types.MethodType(NewGenerationMixin.sample_stream, self._model)
        self._model._sample = types.MethodType(NewGenerationMixin.sample_stream, self._model)

        self._tokenizer = self._processor.tokenizer
        print(self._model)
        print(f"{self.max_position_embeddings=}")

    def get_multimodal_tokens(self, full_prompt, image_paths=None):
        num_tokens = len(self.tokenizer.encode(full_prompt))
        for image_path in image_paths:
            num_tokens += IMAGE_TOKEN_LENGTH * MAX_PACHES
        return num_tokens
    
    def generate_yield_string(self, prompt, temperature, max_tokens, stop_strings: Optional[Tuple[str]] = None, **kwargs):
        from transformers.generation.utils import GenerationConfig
        from PIL import Image
        image_paths = kwargs.get("image_paths", None)
        image_paths = image_paths or []

        images = [Image.open(x) for x in image_paths] if len(image_paths) > 0 else None

        with torch.no_grad():
            inputs = self.processor(prompt, images, return_tensors='pt')
            # inputs = inputs.to("cuda", torch.bfloat16)
            inputs = {k: v.to(self.device_map) for k, v in inputs.items() if v is not None}
            num_tokens = self.get_multimodal_tokens(prompt, image_paths)
            # non-streaming generation
            # output = self._model.generate(
            #     **inputs, 
            #     do_sample=True, 
            #     temperature=temperature, 
            #     max_new_tokens=max_tokens,
            #     pad_token_id=self.processor.tokenizer.pad_token_id,
            # )
            # # response = self.processor.tokenizer.decode(output[0][-inputs.input_ids.size(-1):], skip_special_tokens=True)
            # full_output_text = self.processor.decode(output[0], skip_special_tokens=True)
            # response = full_output_text.split("<|im_start|>assistant\n")[-1]
            # num_tokens = self.get_multimodal_tokens(prompt + response, image_paths)
            # print(prompt)
            # print(response)
            # print(num_tokens)
            # yield response, num_tokens

            # if i % 4 == 0 and i > 1:
            #     message_safety = safety_check(response)
            #     if message_safety is not None:
            #         history = undo_history(history)
            #         yield history, "", None
            #         raise gr.Error(message_safety)

            # # ! streaming
            generator = self._model.generate(
                **inputs, 
                do_sample=True, 
                temperature=temperature, 
                max_new_tokens=max_tokens, 
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

            out_tokens = []
            response = None
            for index, token in enumerate(generator):
                out_tokens.append(token.item())
                response = self.processor.tokenizer.decode(out_tokens)

                yield response, num_tokens
            
            del generator
            
            if response is not None:

                full_text = prompt + response
                num_tokens = self.get_multimodal_tokens(full_text, image_paths)
                yield response, num_tokens

        # raw_image = Image.open(requests.get(image_file, stream=True).raw)
        # inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)




