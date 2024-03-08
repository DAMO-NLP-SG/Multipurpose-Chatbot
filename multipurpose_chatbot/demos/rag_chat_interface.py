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
from gradio.themes import ThemeClass as Theme

from .base_demo import register_demo, get_demo_class, BaseDemo

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


from ..globals import MODEL_ENGINE, RAG_CURRENT_FILE, RAG_EMBED, load_embeddings, get_rag_embeddings

from .chat_interface import (
    SYSTEM_PROMPT,
    MODEL_NAME,
    MAX_TOKENS,
    TEMPERATURE,
    CHAT_EXAMPLES,
    gradio_history_to_openai_conversations,
    gradio_history_to_conversation_prompt,
    DATETIME_FORMAT,
    get_datetime_string,
    format_conversation,
    chat_response_stream_multiturn_engine,
    ChatInterfaceDemo,
    CustomizedChatInterface,
)

from ..configs import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RAG_EMBED_MODEL_NAME,
)

RAG_CURRENT_VECTORSTORE = None


def load_document_split_vectorstore(file_path):
    global RAG_CURRENT_FILE, RAG_EMBED, RAG_CURRENT_VECTORSTORE
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
    from langchain_community.vectorstores import Chroma, FAISS
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    splits = loader.load_and_split(splitter)
    RAG_CURRENT_VECTORSTORE = FAISS.from_texts(texts=[s.page_content for s in splits], embedding=get_rag_embeddings())
    return RAG_CURRENT_VECTORSTORE

def docs_to_context_content(docs: List[Any]):
    content = "\n".join([d.page_content for d in docs])
    return content


DOC_TEMPLATE = """###
{content}
###

"""

DOC_INSTRUCTION = """Answer the following query exclusively based on the information provided in the document above. \
If the information is not found, please say so instead of making up facts! Remember to answer the question in the same language as the user query!
"""


def docs_to_rag_context(docs: List[Any], doc_instruction=None):
    doc_instruction = doc_instruction or DOC_INSTRUCTION
    content = docs_to_context_content(docs)
    context = doc_instruction.strip() + "\n" + DOC_TEMPLATE.format(content=content)
    return context


def maybe_get_doc_context(message, file_input, rag_num_docs: Optional[int] = 3):
    doc_context = None
    if file_input is not None:
        if file_input == RAG_CURRENT_FILE:
            # reuse
            vectorstore = RAG_CURRENT_VECTORSTORE
            print(f'Reuse vectorstore: {file_input}')
        else:
            vectorstore = load_document_split_vectorstore(file_input)
            print(f'New vectorstore: {RAG_CURRENT_FILE} {file_input}')
            RAG_CURRENT_FILE = file_input
        docs = vectorstore.similarity_search(message, k=rag_num_docs)
        doc_context = docs_to_rag_context(docs)
    return doc_context


def chat_response_stream_multiturn_doc_engine(
    message: str, 
    history: List[Tuple[str, str]], 
    file_input: Optional[str] = None,
    temperature: float = 0.7, 
    max_tokens: int = 1024, 
    system_prompt: Optional[str] = SYSTEM_PROMPT,
    rag_num_docs: Optional[int] = 3,
    doc_instruction: Optional[str] = DOC_INSTRUCTION,
    # profile: Optional[gr.OAuthProfile] = None,
):
    global MODEL_ENGINE, RAG_CURRENT_FILE, RAG_EMBED, RAG_CURRENT_VECTORSTORE
    if len(message) == 0:
        raise gr.Error("The message cannot be empty!")
    
    rag_num_docs = int(rag_num_docs)
    doc_instruction = doc_instruction or DOC_INSTRUCTION
    doc_context = None
    if file_input is not None:
        if file_input == RAG_CURRENT_FILE:
            # reuse
            vectorstore = RAG_CURRENT_VECTORSTORE
            print(f'Reuse vectorstore: {file_input}')
        else:
            vectorstore = load_document_split_vectorstore(file_input)
            print(f'New vectorstore: {RAG_CURRENT_FILE} {file_input}')
            RAG_CURRENT_FILE = file_input
        docs = vectorstore.similarity_search(message, k=rag_num_docs)
        # doc_context = docs_to_rag_context(docs)
        rag_content = docs_to_context_content(docs)
        doc_context = doc_instruction.strip() + "\n" + DOC_TEMPLATE.format(content=rag_content)
    
    if doc_context is not None:
        message = f"{doc_context}\n\n{message}"
    
    for response, num_tokens in chat_response_stream_multiturn_engine(
        message, history, temperature, max_tokens, system_prompt
    ):
        # ! yield another content which is doc_context
        yield response, num_tokens, doc_context



class RagChatInterface(CustomizedChatInterface):
    def __init__(
            self, 
            fn: Callable[..., Any], 
            *, 
            chatbot: gr.Chatbot | None = None, 
            textbox: gr.Textbox | None = None, 
            additional_inputs: str | Component | list[str | Component] | None = None, 
            additional_inputs_accordion_name: str | None = None, 
            additional_inputs_accordion: str | gr.Accordion | None = None, 
            render_additional_inputs_fn: Callable | None = None,
            examples: list[str] | None = None, 
            cache_examples: bool | None = None, 
            title: str | None = None, 
            description: str | None = None, 
            theme: Theme | str | None = None, 
            css: str | None = None, 
            js: str | None = None, 
            head: str | None = None, 
            analytics_enabled: bool | None = None, 
            submit_btn: str | Button | None = "Submit", 
            stop_btn: str | Button | None = "Stop", 
            retry_btn: str | Button | None = "🔄  Retry", 
            undo_btn: str | Button | None = "↩️ Undo", 
            clear_btn: str | Button | None = "🗑️  Clear", 
            autofocus: bool = True, 
            concurrency_limit: int | Literal['default'] | None = "default", 
            fill_height: bool = True
        ):
        try:
            super(gr.ChatInterface, self).__init__(
                analytics_enabled=analytics_enabled,
                mode="chat_interface",
                css=css,
                title=title or "Gradio",
                theme=theme,
                js=js,
                head=head,
                fill_height=fill_height,
            )
        except Exception as e:
            # Handling some old gradio version with out fill_height
            super(gr.ChatInterface, self).__init__(
                analytics_enabled=analytics_enabled,
                mode="chat_interface",
                css=css,
                title=title or "Gradio",
                theme=theme,
                js=js,
                head=head,
                # fill_height=fill_height,
            )
        self.concurrency_limit = concurrency_limit
        self.fn = fn
        self.render_additional_inputs_fn = render_additional_inputs_fn
        self.is_async = inspect.iscoroutinefunction(
            self.fn
        ) or inspect.isasyncgenfunction(self.fn)
        self.is_generator = inspect.isgeneratorfunction(
            self.fn
        ) or inspect.isasyncgenfunction(self.fn)
        self.examples = examples
        if self.space_id and cache_examples is None:
            self.cache_examples = True
        else:
            self.cache_examples = cache_examples or False
        self.buttons: list[Button | None] = []

        if additional_inputs:
            if not isinstance(additional_inputs, list):
                additional_inputs = [additional_inputs]
            self.additional_inputs = [
                get_component_instance(i)
                for i in additional_inputs  # type: ignore
            ]
        else:
            self.additional_inputs = []
        if additional_inputs_accordion_name is not None:
            print(
                "The `additional_inputs_accordion_name` parameter is deprecated and will be removed in a future version of Gradio. Use the `additional_inputs_accordion` parameter instead."
            )
            self.additional_inputs_accordion_params = {
                "label": additional_inputs_accordion_name
            }
        if additional_inputs_accordion is None:
            self.additional_inputs_accordion_params = {
                "label": "Additional Inputs",
                "open": False,
            }
        elif isinstance(additional_inputs_accordion, str):
            self.additional_inputs_accordion_params = {
                "label": additional_inputs_accordion
            }
        elif isinstance(additional_inputs_accordion, Accordion):
            self.additional_inputs_accordion_params = (
                additional_inputs_accordion.recover_kwargs(
                    additional_inputs_accordion.get_config()
                )
            )
        else:
            raise ValueError(
                f"The `additional_inputs_accordion` parameter must be a string or gr.Accordion, not {type(additional_inputs_accordion)}"
            )

        with self:
            if title:
                Markdown(
                    f"<h1 style='text-align: center; margin-bottom: 1rem'>{self.title}</h1>"
                )
            if description:
                Markdown(description)

            if chatbot:
                self.chatbot = chatbot.render()
            else:
                self.chatbot = Chatbot(
                    label="Chatbot", scale=1, height=200 if fill_height else None
                )

            with Row():
                for btn in [retry_btn, undo_btn, clear_btn]:
                    if btn is not None:
                        if isinstance(btn, Button):
                            btn.render()
                        elif isinstance(btn, str):
                            btn = Button(btn, variant="secondary", size="sm")
                        else:
                            raise ValueError(
                                f"All the _btn parameters must be a gr.Button, string, or None, not {type(btn)}"
                            )
                    self.buttons.append(btn)  # type: ignore

            with Group():
                with Row():
                    if textbox:
                        textbox.container = False
                        textbox.show_label = False
                        textbox_ = textbox.render()
                        assert isinstance(textbox_, Textbox)
                        self.textbox = textbox_
                    else:
                        self.textbox = Textbox(
                            container=False,
                            show_label=False,
                            label="Message",
                            placeholder="Type a message...",
                            scale=7,
                            autofocus=autofocus,
                        )
                    if submit_btn is not None:
                        if isinstance(submit_btn, Button):
                            submit_btn.render()
                        elif isinstance(submit_btn, str):
                            submit_btn = Button(
                                submit_btn,
                                variant="primary",
                                scale=2,
                                min_width=150,
                            )
                        else:
                            raise ValueError(
                                f"The submit_btn parameter must be a gr.Button, string, or None, not {type(submit_btn)}"
                            )
                    if stop_btn is not None:
                        if isinstance(stop_btn, Button):
                            stop_btn.visible = False
                            stop_btn.render()
                        elif isinstance(stop_btn, str):
                            stop_btn = Button(
                                stop_btn,
                                variant="stop",
                                visible=False,
                                scale=2,
                                min_width=150,
                            )
                        else:
                            raise ValueError(
                                f"The stop_btn parameter must be a gr.Button, string, or None, not {type(stop_btn)}"
                            )
                    self.num_tokens = Textbox(
                            container=False,
                            label="num_tokens",
                            placeholder="0 tokens",
                            scale=1,
                            interactive=False,
                            # autofocus=autofocus,
                            min_width=10
                        )
                    self.buttons.extend([submit_btn, stop_btn])  # type: ignore
                
                self.fake_api_btn = Button("Fake API", visible=False)
                self.fake_response_textbox = Textbox(label="Response", visible=False)
                (
                    self.retry_btn,
                    self.undo_btn,
                    self.clear_btn,
                    self.submit_btn,
                    self.stop_btn,
                ) = self.buttons

            if examples:
                if self.is_generator:
                    examples_fn = self._examples_stream_fn
                else:
                    examples_fn = self._examples_fn

                self.examples_handler = Examples(
                    examples=examples,
                    inputs=[self.textbox] + self.additional_inputs,
                    outputs=self.chatbot,
                    fn=examples_fn,
                )

            any_unrendered_inputs = any(
                not inp.is_rendered for inp in self.additional_inputs
            )
            if self.additional_inputs and any_unrendered_inputs:
                with Accordion(**self.additional_inputs_accordion_params):  # type: ignore
                    if self.render_additional_inputs_fn is not None:
                        self.render_additional_inputs_fn()
                    else:
                        for input_component in self.additional_inputs:
                            if not input_component.is_rendered:
                                input_component.render()
            
            self.rag_content = gr.Textbox(
                scale=4,
                lines=16,
                label='Retrieved RAG context',
                placeholder="Rag context and instrution will show up here",
                interactive=False
            )

            # The example caching must happen after the input components have rendered
            if cache_examples:
                client_utils.synchronize_async(self.examples_handler.cache)

            self.saved_input = State()
            self.chatbot_state = (
                State(self.chatbot.value) if self.chatbot.value else State([])
            )

            self._setup_events()
            self._setup_api()
    
    def _setup_events(self) -> None:
        from gradio.components import State
        has_on = False
        try:
            from gradio.events import Dependency, EventListenerMethod, on
            has_on = True
        except ImportError as ie:
            has_on = False
        submit_fn = self._stream_fn if self.is_generator else self._submit_fn
        if not self.is_generator:
            raise NotImplementedError(f'should use generator')

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
                    [self.chatbot, self.chatbot_state, self.num_tokens, self.rag_content],
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
                    [self.chatbot, self.chatbot_state, self.num_tokens, self.rag_content],
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

        # ! In case of error, yield the previous history & undo any generation before raising error
        try:
            first_response_pack = await async_iteration(generator)
            if isinstance(first_response_pack, (tuple, list)):
                first_response, num_tokens, rag_content = first_response_pack
            else:
                first_response, num_tokens, rag_content = first_response_pack, -1, ""
            update = history + [[message, first_response]]
            yield update, update, f"{num_tokens} toks", rag_content
        except StopIteration:
            update = history + [[message, None]]
            yield update, update, "NaN toks", ""
        except Exception as e:
            yield history, history, "NaN toks", ""
            raise e

        try:
            async for response_pack in generator:
                if isinstance(response_pack, (tuple, list)):
                    response, num_tokens, rag_content = response_pack
                else:
                    response, num_tokens, rag_content = response_pack, "NaN toks", ""
                update = history + [[message, response]]
                yield update, update, f"{num_tokens} toks", rag_content
        except Exception as e:
            yield history, history, "NaN toks", ""
            raise e



@register_demo
class RagChatInterfaceDemo(ChatInterfaceDemo):

    @property
    def examples(self):
        return [
            ["Explain how attention works.", "assets/attention_all_you_need.pdf"],
            ["Explain why the sky is blue.", None],
        ]
    
    @property
    def tab_name(self):
        return "RAG Chat"

    def create_demo(
            self, 
            title: str | None = None, 
            description: str | None = None, 
            **kwargs
        ) -> gr.Blocks:
        load_embeddings()
        global RAG_EMBED
        # assert RAG_EMBED is not None
        print(F'{RAG_EMBED=}')
        system_prompt = kwargs.get("system_prompt", SYSTEM_PROMPT)
        max_tokens = kwargs.get("max_tokens", MAX_TOKENS)
        temperature = kwargs.get("temperature", TEMPERATURE)
        model_name = kwargs.get("model_name", MODEL_NAME)
        rag_num_docs = kwargs.get("rag_num_docs", 3)

        from ..configs import RAG_EMBED_MODEL_NAME

        description = (
            description or 
            f"""Upload a long document to ask question with RAG. Check at the bottom the retrieved RAG text segment. 
Control `RAG instruction to fit your language`. Embedding model {RAG_EMBED_MODEL_NAME}."""
        )

        additional_inputs = [
            gr.File(label='Upload Document', file_count='single', file_types=['pdf', 'docx', 'txt']),
            gr.Number(value=temperature, label='Temperature', min_width=20), 
            gr.Number(value=max_tokens, label='Max tokens', min_width=20), 
            gr.Textbox(value=system_prompt, label='System prompt', lines=2),
            gr.Number(value=rag_num_docs, label='RAG Top-K', min_width=20),
            gr.Textbox(value=DOC_INSTRUCTION, label='RAG instruction'),
        ]
        def render_additional_inputs_fn():
            additional_inputs[0].render()
            with Row():
                additional_inputs[1].render()
                additional_inputs[2].render()
                additional_inputs[4].render()
            additional_inputs[3].render()
            additional_inputs[5].render()

        demo_chat = RagChatInterface(
            chat_response_stream_multiturn_doc_engine,
            chatbot=gr.Chatbot(
                label=model_name,
                bubble_full_width=False,
                latex_delimiters=[
                    { "left": "$", "right": "$", "display": False},
                    { "left": "$$", "right": "$$", "display": True},
                ],
                show_copy_button=True,
            ),
            textbox=gr.Textbox(placeholder='Type message', lines=1, max_lines=128, min_width=200, scale=8),
            submit_btn=gr.Button(value='Submit', variant="primary", scale=0),
            # ! consider preventing the stop button
            # stop_btn=None,
            title=title,
            description=description,
            additional_inputs=additional_inputs, 
            render_additional_inputs_fn=render_additional_inputs_fn,
            additional_inputs_accordion=gr.Accordion("Additional Inputs", open=True),
            examples=self.examples,
            cache_examples=False,
        )
        return demo_chat
    

