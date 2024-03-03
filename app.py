# Copyright: DAMO Academy, Alibaba Group
# By Xuan Phi Nguyen at DAMO Academy, Alibaba Group

# Description:
"""
Demo script to launch Language chat model for Southeast Asian Languages
"""


import os
from gradio.themes import ThemeClass as Theme
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
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from gradio.components import Button, Component
from gradio.events import Dependency, EventListenerMethod


# @@ environments ================

DTYPE = os.environ.get("DTYPE", "bfloat16")

# ! uploaded model path, will be downloaded to MODEL_PATH
HF_MODEL_NAME = os.environ.get("HF_MODEL_NAME", "DAMO-NLP-SG/seal-13b-chat-a")
# ! if model is private, need HF_TOKEN to access the model
HF_TOKEN = os.environ.get("HF_TOKEN", None)
# ! path where the model is downloaded, either on ./ or persistent disc
MODEL_PATH = os.environ.get("MODEL_PATH", "./seal-13b-chat-a")

MODEL_NAME = str(os.environ.get("MODEL_NAME", "SeaLLM-7B"))

# gradio config
PORT = int(os.environ.get("PORT", "7860"))
# how many iterations to yield response
STREAM_YIELD_MULTIPLE = int(os.environ.get("STREAM_YIELD_MULTIPLE", "1"))
# how many iterations to perform safety check on response
STREAM_CHECK_MULTIPLE = int(os.environ.get("STREAM_CHECK_MULTIPLE", "0"))

# whether to enable to popup accept user
ENABLE_AGREE_POPUP = bool(int(os.environ.get("ENABLE_AGREE_POPUP", "0")))

# self explanatory
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2048"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.1"))
FREQUENCE_PENALTY = float(os.environ.get("FREQUENCE_PENALTY", "0.1"))
PRESENCE_PENALTY = float(os.environ.get("PRESENCE_PENALTY", "0.0"))

# whether to enable quantization, currently not in use
QUANTIZATION = str(os.environ.get("QUANTIZATION", ""))

# Batch inference file upload
ENABLE_BATCH_INFER = bool(int(os.environ.get("ENABLE_BATCH_INFER", "1")))
BATCH_INFER_MAX_ITEMS = int(os.environ.get("BATCH_INFER_MAX_ITEMS", "100"))
BATCH_INFER_MAX_FILE_SIZE = int(os.environ.get("BATCH_INFER_MAX_FILE_SIZE", "500"))
BATCH_INFER_MAX_PROMPT_TOKENS = int(os.environ.get("BATCH_INFER_MAX_PROMPT_TOKENS", "4000"))
BATCH_INFER_SAVE_TMP_FILE = os.environ.get("BATCH_INFER_SAVE_TMP_FILE", "./tmp/pred.json")



allowed_paths = []



# @@ constants ================

DTYPES = {
    'float16': torch.float16,
    'bfloat16': torch.bfloat16
}

llm = None
demo = None


SYSTEM_PROMPT_1 = """You are a helpful, respectful, honest and safe AI assistant."""
# SYSTEM_PROMPT_1 = """You are a helpful, respectful, honest and safe AI assistant. Please use the following real time information if requested.
# Current date and time: {cur_datetime}."""

# ============ CONSTANT ============
# https://github.com/gradio-app/gradio/issues/884

# MODEL_TITLE = """
# <div class="container" style="
#     align-items: center;
#     justify-content: center;
#     display: flex;
# ">  
#     <div class="image" >
#         <img src="file/seal_logo.png" style="
#             max-width: 10em;
#             max-height: 5%;
#             height: 3em;
#             width: 3em;
#             float: left;
#             margin-left: auto;
#         ">
#     </div>
#     <div class="text" style="
#         padding-left: 20px;
#         padding-top: 1%;
#         float: left;
#     ">
#         <h1 style="font-size: xx-large">SeaLLMs - Large Language Models for Southeast Asia</h1>
#     </div>
# </div>
# """

MODEL_TITLE = "<h1>Multi-Purpose Chatbot</h1>"


MODEL_DESC = f"""
<div style='display:flex; gap: 0.25rem; '>
<a href='https://github.com/DAMO-NLP-SG/Multipurpose-Chatbot'><img src='https://img.shields.io/badge/Github-Code-success'></a>
</div>
<span style="font-size: larger">
A multi-purpose helpful assistant with multiple functionalities (Chat, Freeform, batch, RAG...).
</span>
""".strip()


path_markdown = """
<h4>Model Name: {model_path}</h4>
"""


cite_markdown = """
## Citation
If you find our project useful, hope you can star our repo and cite our paper as follows:
```
@article{damonlpsg2023seallm,
  author = {Xuan-Phi Nguyen*, Wenxuan Zhang*, Xin Li*, Mahani Aljunied*, Zhiqiang Hu, Chenhui Shen^, Yew Ken Chia^, Xingxuan Li, Jianyu Wang, Qingyu Tan, Liying Cheng, Guanzheng Chen, Yue Deng, Sen Yang, Chaoqun Liu, Hang Zhang, Lidong Bing},
  title = {SeaLLMs - Large Language Models for Southeast Asia},
  year = 2023,
}
```
"""



# ! ==================================================================

set_documentation_group("component")


MODEL_ENGINE = None
RAG_EMBED = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'trust_remote_code':True})
RAG_CURRENT_FILE = None
RAG_CURRENT_VECTORSTORE = None


@document()
class ChatBot(gr.Chatbot):
    def _postprocess_chat_messages(
        self, chat_message
    ):
        x = super()._postprocess_chat_messages(chat_message)
        return x


# replace events so that submit button is disabled during generation, if stop_btn not found
# this prevent weird behavior
def _setup_stop_events(
    self, event_triggers: list[EventListenerMethod], event_to_cancel: Dependency
) -> None:
    from gradio.components import State
    event_triggers = event_triggers if isinstance(event_triggers, (list, tuple)) else [event_triggers]
    if self.stop_btn and self.is_generator:
        if self.submit_btn:
            for event_trigger in event_triggers:
                event_trigger(
                    lambda: (
                        Button(visible=False),
                        Button(visible=True),
                    ),
                    None,
                    [self.submit_btn, self.stop_btn],
                    api_name=False,
                    queue=False,
                )
            event_to_cancel.then(
                lambda: (Button(visible=True), Button(visible=False)),
                None,
                [self.submit_btn, self.stop_btn],
                api_name=False,
                queue=False,
            )
        else:
            for event_trigger in event_triggers:
                event_trigger(
                    lambda: Button(visible=True),
                    None,
                    [self.stop_btn],
                    api_name=False,
                    queue=False,
                )
            event_to_cancel.then(
                lambda: Button(visible=False),
                None,
                [self.stop_btn],
                api_name=False,
                queue=False,
            )
        self.stop_btn.click(
            None,
            None,
            None,
            cancels=event_to_cancel,
            api_name=False,
        )
    else:
        if self.submit_btn:
            for event_trigger in event_triggers:
                event_trigger(
                    lambda: Button(interactive=False),
                    None,
                    [self.submit_btn],
                    api_name=False,
                    queue=False,
                )
            event_to_cancel.then(
                lambda: Button(interactive=True),
                None,
                [self.submit_btn],
                api_name=False,
                queue=False,
            )
    # upon clear, cancel the submit event as well
    if self.clear_btn:
        self.clear_btn.click(
            lambda: ([], [], None, Button(interactive=True)),
            None,
            [self.chatbot, self.chatbot_state, self.saved_input, self.submit_btn],
            queue=False,
            api_name=False,
            cancels=event_to_cancel,
        )

# TODO: reconfigure clear button as stop and clear button
def _setup_events(self) -> None:
    from gradio.components import State
    has_on = False
    try:
        from gradio.events import Dependency, EventListenerMethod, on
        has_on = True
    except ImportError as ie:
        has_on = False
    submit_fn = self._stream_fn if self.is_generator else self._submit_fn

    if has_on:
        # new version
        submit_triggers = (
            [self.textbox.submit, self.submit_btn.click]
            if self.submit_btn
            else [self.textbox.submit]
        )
        submit_event = (
            on(
                submit_triggers,
                self._clear_and_save_textbox,
                [self.textbox],
                [self.textbox, self.saved_input],
                api_name=False,
                queue=False,
            )
            .then(
                self._display_input,
                [self.saved_input, self.chatbot_state],
                [self.chatbot, self.chatbot_state],
                api_name=False,
                queue=False,
            )
            .then(
                submit_fn,
                [self.saved_input, self.chatbot_state] + self.additional_inputs,
                [self.chatbot, self.chatbot_state],
                api_name=False,
            )
        )
        self._setup_stop_events(submit_triggers, submit_event)
    else:
        raise ValueError(f'Better install new gradio version than 3.44.0')

    if self.retry_btn:
        retry_event = (
            self.retry_btn.click(
                self._delete_prev_fn,
                [self.chatbot_state],
                [self.chatbot, self.saved_input, self.chatbot_state],
                api_name=False,
                queue=False,
            )
            .then(
                self._display_input,
                [self.saved_input, self.chatbot_state],
                [self.chatbot, self.chatbot_state],
                api_name=False,
                queue=False,
            )
            .then(
                submit_fn,
                [self.saved_input, self.chatbot_state] + self.additional_inputs,
                [self.chatbot, self.chatbot_state],
                api_name=False,
            )
        )
        self._setup_stop_events([self.retry_btn.click], retry_event)

    if self.undo_btn:
        self.undo_btn.click(
            self._delete_prev_fn,
            [self.chatbot_state],
            [self.chatbot, self.saved_input, self.chatbot_state],
            api_name=False,
            queue=False,
        ).then(
            lambda x: x,
            [self.saved_input],
            [self.textbox],
            api_name=False,
            queue=False,
        )

    # Reconfigure clear_btn to stop and clear text box


def _display_input(
        self, message: str, history: List[List[Union[str, None]]]
    ) -> Tuple[List[List[Union[str, None]]], List[List[list[Union[str, None]]]]]:
    if message is not None and message.strip() != "":
        history.append([message, None])
    return history, history


async def _stream_fn(
    self,
    message: str,
    history_with_input,
    request: Request,
    *args,
) -> AsyncGenerator:
    history = history_with_input[:-1]
    inputs, _, _ = special_args(
        self.fn, inputs=[message, history, *args], request=request
    )

    if self.is_async:
        generator = self.fn(*inputs)
    else:
        generator = await anyio.to_thread.run_sync(
            self.fn, *inputs, limiter=self.limiter
        )
        generator = SyncToAsyncIterator(generator, self.limiter)
    try:
        first_response = await async_iteration(generator)
        update = history + [[message, first_response]]
        yield update, update
    except StopIteration:
        update = history + [[message, None]]
        yield update, update
    except Exception as e:
        yield history, history
        raise e

    try:
        async for response in generator:
            update = history + [[message, response]]
            yield update, update
    except Exception as e:
        yield history, history
        raise e


# replace
gr.ChatInterface._setup_stop_events = _setup_stop_events
gr.ChatInterface._setup_events = _setup_events
gr.ChatInterface._display_input = _display_input
gr.ChatInterface._stream_fn = _stream_fn


@document()
class CustomTabbedInterface(gr.Blocks):
    def __init__(
        self,
        interface_list: list[gr.Interface],
        tab_names: Optional[list[str]] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        theme: Optional[gr.Theme] = None,
        analytics_enabled: Optional[bool] = None,
        css: Optional[str] = None,
    ):
        """
        Parameters:
            interface_list: a list of interfaces to be rendered in tabs.
            tab_names: a list of tab names. If None, the tab names will be "Tab 1", "Tab 2", etc.
            title: a title for the interface; if provided, appears above the input and output components in large font. Also used as the tab title when opened in a browser window.
            analytics_enabled: whether to allow basic telemetry. If None, will use GRADIO_ANALYTICS_ENABLED environment variable or default to True.
            css: custom css or path to custom css file to apply to entire Blocks
        Returns:
            a Gradio Tabbed Interface for the given interfaces
        """
        super().__init__(
            title=title or "Gradio",
            theme=theme,
            analytics_enabled=analytics_enabled,
            mode="tabbed_interface",
            css=css,
        )
        self.description = description
        if tab_names is None:
            tab_names = [f"Tab {i}" for i in range(len(interface_list))]
        with self:
            if title:
                gr.Markdown(
                    f"<h1 style='text-align: center; margin-bottom: 1rem'>{title}</h1>"
                )
            if description:
                gr.Markdown(description)
            with gr.Tabs():
                for interface, tab_name in zip(interface_list, tab_names):
                    with gr.Tab(label=tab_name):
                        interface.render()


@document()
class DocChatInterface(gr.ChatInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with self:
            file_input = gr.File(label="Upload document", file_count='single', file_types=['pdf', 'docx', 'txt', 'json'])
        self.additional_inputs = self.additional_inputs + [file_input]




# ! avoid saying 




def gradio_history_to_openai_conversations(message, history=None, system_prompt=None):
    conversations = []
    system_prompt = system_prompt or "You are a helpful assistant."
    if history is not None and len(history) > 0:
        for i, (prompt, res) in enumerate(history):
            if prompt is not None:
                conversations.append({"role": "user", "content": prompt.strip()})
            if res is not None:
                conversations.append({"role": "assistant", "content": res.strip()})
    conversations.append({"role": "user", "content": message.strip()})
    if conversations[0]['role'] != 'system':
        conversations = [{"role": "system", "content": system_prompt}] + conversations
    return conversations


def gradio_history_to_conversation_prompt(message, history=None, system_prompt=None):
    global MODEL_ENGINE
    full_prompt = MODEL_ENGINE.apply_chat_template(
        gradio_history_to_openai_conversations(
            message.strip(), history=history, system_prompt=system_prompt),
        add_generation_prompt=True
    )
    return full_prompt


DATETIME_FORMAT = "Current date time: {cur_datetime}."
def get_datetime_string():
    from datetime import datetime
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%B %d, %Y, %H:%M:%S")
    return dt_string

    
def chat_response_stream_multiturn_engine(
    message: str, 
    history: List[Tuple[str, str]], 
    temperature: float, 
    max_tokens: int, 
    # frequency_penalty: float,
    # presence_penalty: float,
    system_prompt: Optional[str] = SYSTEM_PROMPT_1,
    # file_input: Optional[str] = None,
    # profile: Optional[gr.OAuthProfile] = None,
) -> str:
    global MODEL_ENGINE
    temperature = float(temperature)
    # ! remove frequency_penalty
    # frequency_penalty = float(frequency_penalty)
    max_tokens = int(max_tokens)
    # print(f"{file_input=}")
    message = message.strip()
    if len(message) == 0:
        raise gr.Error("The message cannot be empty!")
    # ! skip safety
    if DATETIME_FORMAT in system_prompt:
        # ! This sometime works sometimes dont
        system_prompt = system_prompt.format(cur_datetime=get_datetime_string())
    # full_prompt = chatml_format(message.strip(), history=history, system_prompt=system_prompt)
    full_prompt = gradio_history_to_conversation_prompt(message.strip(), history=history, system_prompt=system_prompt)
    # ! skip length checked
    # if len(tokenizer.encode(full_prompt)) >= 4050:
    #     raise gr.Error(f"Conversation or prompt is too long, please clear the chatbox or try shorter input.")
    print(full_prompt)
    ostring = None
    for j, ostring in enumerate(MODEL_ENGINE.generate_yield_string(
        prompt=full_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )):
        yield ostring
    
    history_str = format_conversation(history + [[message, ostring]])
    print(f'@@@@@@@@@@\n{history_str}\n##########\n')

    if ostring is not None:
        yield ostring


def load_document_split_vectorstore(file_path):
    global RAG_CURRENT_FILE, RAG_EMBED, RAG_CURRENT_VECTORSTORE
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)

    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)

    splits = loader.load_and_split(splitter)
    RAG_CURRENT_VECTORSTORE = Chroma.from_documents(documents=splits, embedding=RAG_EMBED)
    return RAG_CURRENT_VECTORSTORE
    

def docs_to_rag_context(docs: List[str]):
    contexts = "\n".join([d.page_content for d in docs])
    context = f"""### Begin document
{contexts}
### End document
Asnwer the following query exclusively based on the information provided in the document above. \
Remember to follow the language of the user query.
"""
    return context


def chat_response_stream_multiturn_doc_engine(
    message: str, 
    history: List[Tuple[str, str]], 
    file_input: Optional[str] = None,
    temperature: float = 0.7, 
    max_tokens: int = 1024, 
    # frequency_penalty: float,
    # presence_penalty: float,
    system_prompt: Optional[str] = SYSTEM_PROMPT_1,
    # profile: Optional[gr.OAuthProfile] = None,
) -> str:
    print(f'{file_input=}')
    global MODEL_ENGINE, RAG_CURRENT_FILE, RAG_EMBED, RAG_CURRENT_VECTORSTORE
    num_docs = 3
    
    doc_context = None
    if file_input is not None:
        if file_input == RAG_CURRENT_FILE:
            # reuse
            vectorstore = RAG_CURRENT_VECTORSTORE
            print(f'Reuse vectorstore')
        else:
            vectorstore = load_document_split_vectorstore(file_input)
            print(f'New vectorstore')
            RAG_CURRENT_FILE = file_input
        docs = vectorstore.similarity_search(message, k=num_docs)
        doc_context = docs_to_rag_context(docs)
    
    print(F"{doc_context=}")

    temperature = float(temperature)
    # ! remove frequency_penalty
    # frequency_penalty = float(frequency_penalty)
    max_tokens = int(max_tokens)
    # print(f"{file_input=}")
    message = message.strip()

    if doc_context is not None:
        message = f"{doc_context}\n\n{message}"

    if len(message) == 0:
        raise gr.Error("The message cannot be empty!")
    # ! skip safety
    if DATETIME_FORMAT in system_prompt:
        # ! This sometime works sometimes dont
        system_prompt = system_prompt.format(cur_datetime=get_datetime_string())
    # full_prompt = chatml_format(message.strip(), history=history, system_prompt=system_prompt)
    full_prompt = gradio_history_to_conversation_prompt(message.strip(), history=history, system_prompt=system_prompt)
    # ! skip length checked
    # if len(tokenizer.encode(full_prompt)) >= 4050:
    #     raise gr.Error(f"Conversation or prompt is too long, please clear the chatbox or try shorter input.")
    print(full_prompt)
    ostring = None
    for j, ostring in enumerate(MODEL_ENGINE.generate_yield_string(
        prompt=full_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )):
        yield ostring
    
    history_str = format_conversation(history + [[message, ostring]])
    print(f'@@@@@@@@@@\n{history_str}\n##########\n')

    if ostring is not None:
        yield ostring

def generate_free_form_stream_engine(
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
    if len(message) == 0:
        raise gr.Error("The message cannot be empty!")
    for j, ostring in enumerate(MODEL_ENGINE.generate_yield_string(
        prompt=message,
        temperature=temperature,
        max_tokens=max_tokens,
        stop_strings=stop_strings,
    )):
        yield message + ostring

    if ostring is not None:
        yield message + ostring
    else:
        yield message




def format_conversation(history):
    _str = '\n'.join([
        (
            f'<<<User>>> {h[0]}\n'
            f'<<<Asst>>> {h[1]}'
        )
        for h in history
    ])
    return _str


AGREE_POP_SCRIPTS = """
async () => {
    alert("To use our service, you are required to agree to the following terms:\\nYou must not use our service to generate any harmful, unethical or illegal content that violates local and international laws, including but not limited to hate speech, violence and deception.\\nThe service may collect user dialogue data for performance improvement, and reserves the right to distribute it under CC-BY or similar license. So do not enter any personal information!");
}
"""



def validate_file_item(filename, index, item: Dict[str, str]):
    """
    check safety for items in files
    """
    message = item['prompt'].strip()

    if len(message) == 0:
        raise gr.Error(f'Prompt {index} empty')
    
    
    # tokenizer = llm.get_tokenizer() if llm is not None else None
    # if tokenizer is None or len(tokenizer.encode(message)) >= BATCH_INFER_MAX_PROMPT_TOKENS:
    #     raise gr.Error(f"Prompt {index} too long, should be less than {BATCH_INFER_MAX_PROMPT_TOKENS} tokens")


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
        # frequency_penalty: float,
        # presence_penalty: float,
        stop_strings: str = "[STOP],<s>,</s>,<|im_start|>",
        # current_time: Optional[float] = None,
        system_prompt: Optional[str] = SYSTEM_PROMPT_1
):
    """
    Handle file upload batch inference
    
    """
    global MODEL_ENGINE
    stop_strings = [x.strip() for x in stop_strings.strip().split(",")]
    temperature = float(temperature)
    max_tokens = int(max_tokens)
    all_items, filenames = read_validate_json_files(files)
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
    responses = MODEL_ENGINE.batch_generate(
        full_prompts, temperature=temperature, max_tokens=max_tokens,
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
    return save_path, print_items
    

# BATCH_INFER_MAX_ITEMS
FILE_UPLOAD_DESCRIPTION = f"""Upload JSON file as list of dict with < {BATCH_INFER_MAX_ITEMS} items, \
each item has `prompt` key. We put guardrails to enhance safety, so do not input any harmful content or personal information! Re-upload the file after every submit. See the examples below.
```
[ {{"id": 0, "prompt": "Hello world"}} ,  {{"id": 1, "prompt": "Hi there?"}}]
```
"""

CHAT_EXAMPLES = [
    ["Hãy giải thích thuyết tương đối rộng."],
    ["Tolong bantu saya menulis email ke lembaga pemerintah untuk mencari dukungan finansial untuk penelitian AI."],
    ["แนะนำ 10 จุดหมายปลายทางในกรุงเทพฯ"],
]


# performance items

def create_free_form_generation_demo():
    global short_model_path
    max_tokens = MAX_TOKENS // 2
    temperature = TEMPERATURE
    frequence_penalty = FREQUENCE_PENALTY
    presence_penalty = PRESENCE_PENALTY
    introduction = """
### Free-form | Put any context string (like few-shot prompts)
    """

    with gr.Blocks() as demo_free_form:
        gr.Markdown(introduction)

        with gr.Row():
            txt = gr.Textbox(
                scale=4,
                lines=16,
                show_label=False,
                placeholder="Enter any free form text and submit",
                container=False,
            )
        with gr.Row():
            free_submit_button = gr.Button('Submit', variant='primary')
            free_stop_button = gr.Button('Stop', variant='stop', visible=False)
        with gr.Row():
            temp = gr.Number(value=temperature, label='Temperature', info="Higher -> more random")
            length = gr.Number(value=max_tokens, label='Max tokens', info='Increase if want more generation')
            # freq_pen = gr.Number(value=frequence_penalty, label='Frequency penalty', info='> 0 encourage new tokens over repeated tokens')
            # pres_pen = gr.Number(value=presence_penalty, label='Presence penalty', info='> 0 encourage new tokens, < 0 encourage existing tokens')
            stop_strings = gr.Textbox(value="<s>,</s>,<|im_start|>", label='Stop strings', info='Comma-separated string to stop generation only in FEW-SHOT mode', lines=1)
        
        examples = gr.Examples(
            examples=[
                ["The following is the recite the declaration of independence:",]
            ],
            inputs=[txt, temp, length, stop_strings],
            # outputs=[txt]
        )

        submit_event = free_submit_button.click(
            # generate_free_form_stream, 
            generate_free_form_stream_engine, 
            [txt, temp, length, stop_strings], 
            txt
        )
        # setup stop
        submit_trigger = free_submit_button.click
        submit_trigger(
            lambda: (
                Button(visible=False),
                Button(visible=True),
            ),
            None,
            [free_submit_button, free_stop_button],
            api_name=False,
            queue=False,
        )
        submit_event.then(
            lambda: (Button(visible=True), Button(visible=False)),
            None,
            [free_submit_button, free_stop_button],
            api_name=False,
            queue=False,
        )
        free_stop_button.click(
            None,
            None,
            None,
            cancels=submit_event,
            api_name=False,
        )

    return demo_free_form



def create_file_upload_demo():
    temperature = TEMPERATURE
    frequence_penalty = FREQUENCE_PENALTY
    presence_penalty = PRESENCE_PENALTY
    max_tokens = MAX_TOKENS
    demo_file_upload = gr.Interface(
        # batch_inference,
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
            gr.Textbox(value=SYSTEM_PROMPT_1, label='System prompt', )
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
            ["assets/upload_chat.json", "chat", 0.2, 1024, "<s>,</s>,<|im_start|>"],
            ["assets/upload_few_shot.json", "few-shot", 0.2, 128, "<s>,</s>,<|im_start|>,\\n"]
        ],
        cache_examples=False,
    )
    return demo_file_upload


def create_chat_demo(title=None, description=None):
    sys_prompt = SYSTEM_PROMPT_1
    max_tokens = MAX_TOKENS
    temperature = TEMPERATURE
    # frequence_penalty = FREQUENCE_PENALTY
    # presence_penalty = PRESENCE_PENALTY

    demo_chat = gr.ChatInterface(
        chat_response_stream_multiturn_engine,
        chatbot=ChatBot(
            label=MODEL_NAME,
            bubble_full_width=False,
            latex_delimiters=[
                { "left": "$", "right": "$", "display": False},
                { "left": "$$", "right": "$$", "display": True},
            ],
            show_copy_button=True,
        ),
        # textbox=gr.Textbox(placeholder='Type message', lines=4, max_lines=128, min_width=200),
        textbox=gr.Textbox(placeholder='Type message', lines=1, max_lines=128, min_width=200),
        submit_btn=gr.Button(value='Submit', variant="primary", scale=0),
        # ! consider preventing the stop button
        # stop_btn=None,
        title=title,
        description=description,
        additional_inputs=[
            gr.Number(value=temperature, label='Temperature (higher -> more random)'), 
            gr.Number(value=max_tokens, label='Max generated tokens (increase if want more generation)'), 
            # gr.Number(value=frequence_penalty, label='Frequency penalty (> 0 encourage new tokens over repeated tokens)'), 
            # gr.Number(value=presence_penalty, label='Presence penalty (> 0 encourage new tokens, < 0 encourage existing tokens)'), 
            # gr.Number(value=0, label='current_time', visible=False), 
            # ! Remove the system prompt textbox to avoid jailbreaking
            gr.Textbox(value=sys_prompt, label='System prompt', lines=4)
        ], 
        examples=CHAT_EXAMPLES,
        cache_examples=False
    )
    return demo_chat


def create_rag_chat_demo(title=None, description=None):
    sys_prompt = SYSTEM_PROMPT_1
    max_tokens = MAX_TOKENS
    temperature = TEMPERATURE
    # frequence_penalty = FREQUENCE_PENALTY
    # presence_penalty = PRESENCE_PENALTY

    # demo_chat = DocChatInterface(
    demo_chat = gr.ChatInterface(
        chat_response_stream_multiturn_doc_engine,
        chatbot=ChatBot(
            label=MODEL_NAME,
            bubble_full_width=False,
            latex_delimiters=[
                { "left": "$", "right": "$", "display": False},
                { "left": "$$", "right": "$$", "display": True},
            ],
            show_copy_button=True,
        ),
        # textbox=gr.Textbox(placeholder='Type message', lines=4, max_lines=128, min_width=200),
        textbox=gr.Textbox(placeholder='Type message', lines=1, max_lines=128, min_width=200),
        submit_btn=gr.Button(value='Submit', variant="primary", scale=0),
        # ! consider preventing the stop button
        # stop_btn=None,
        title=title,
        description=description,
        additional_inputs=[
            gr.File(label='Upload Document', file_count='single', file_types=['pdf', 'docx', 'txt', 'json']),
            gr.Number(value=temperature, label='Temperature (higher -> more random)'), 
            gr.Number(value=max_tokens, label='Max generated tokens (increase if want more generation)'), 
            # gr.Number(value=frequence_penalty, label='Frequency penalty (> 0 encourage new tokens over repeated tokens)'), 
            # gr.Number(value=presence_penalty, label='Presence penalty (> 0 encourage new tokens, < 0 encourage existing tokens)'), 
            # gr.Number(value=0, label='current_time', visible=False), 
            # ! Remove the system prompt textbox to avoid jailbreaking
            gr.Textbox(value=sys_prompt, label='System prompt', lines=4),
        ], 
        additional_inputs_accordion=gr.Accordion("Additional Inputs", open=True),
        # examples=CHAT_EXAMPLES,
        cache_examples=False,
    )
    # with demo_chat:
    #     # file_input = gr.File(file_count='single', file_types=['json', ])
    #     file_input = gr.File(file_count='single', file_types=['pdf', 'docx', 'txt', 'json'])

    return demo_chat


def launch_demo():
    global demo, MODEL_ENGINE
    model_desc = MODEL_DESC
    model_path = MODEL_PATH
    model_title = MODEL_TITLE
    hf_model_name = HF_MODEL_NAME
    dtype = DTYPE
    sys_prompt = SYSTEM_PROMPT_1
    max_tokens = MAX_TOKENS
    temperature = TEMPERATURE
    frequence_penalty = FREQUENCE_PENALTY
    presence_penalty = PRESENCE_PENALTY
    ckpt_info = "None"

    print(
        f'Launch config: '
        f'\n| model_title=`{model_title}` '
        f'\n| max_tokens={max_tokens} '
        f'\n| dtype={dtype} '
        f'\n| STREAM_YIELD_MULTIPLE={STREAM_YIELD_MULTIPLE} '
        f'\n| STREAM_CHECK_MULTIPLE={STREAM_CHECK_MULTIPLE} '
        f'\n| frequence_penalty={frequence_penalty} '
        f'\n| presence_penalty={presence_penalty} '
        f'\n| temperature={temperature} '
        f'\n| model_path={model_path} '
        f'\n| Desc={model_desc}'
    )

    from engines.base_engine import BaseEngine
    from engines.mlx_engine import MlxEngine
    MODEL_ENGINE = MlxEngine()
    MODEL_ENGINE.load_model()

    demo_chat = create_chat_demo()
    demo_file_upload = create_file_upload_demo()
    demo_free_form = create_free_form_generation_demo()
    demo_rag_chat = create_rag_chat_demo()

    descriptions = model_desc
    # if DISPLAY_MODEL_PATH:
    descriptions += f"<br> {path_markdown.format(model_path=model_path)}"

    demo = CustomTabbedInterface(
        interface_list=[
            demo_chat, 
            demo_free_form,
            demo_rag_chat,
            demo_file_upload, 
        ],
        tab_names=[
            "Chat Interface", 
            "Completion",
            "RAG Chat Interface", 
            "Batch Inference", 
        ],
        title=f"{model_title}",
        description=descriptions,
    )


    demo.title = MODEL_NAME
    
    with demo:
        gr.Markdown(cite_markdown)
        

    demo.queue(api_open=False)
    return demo




if __name__ == "__main__":
    demo = launch_demo()
    # demo.launch(port=PORT, show_api=False, allowed_paths=allowed_paths)
    demo.launch(show_api=False, allowed_paths=allowed_paths)