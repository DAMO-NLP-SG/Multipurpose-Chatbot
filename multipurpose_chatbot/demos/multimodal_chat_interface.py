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
from gradio.components.base import Component

from .base_demo import register_demo, get_demo_class, BaseDemo


from .chat_interface import (
    SYSTEM_PROMPT,
    MODEL_NAME,
    MAX_TOKENS,
    TEMPERATURE,
    CHAT_EXAMPLES,
    format_conversation,
    gradio_history_to_openai_conversations,
    gradio_history_to_conversation_prompt,
    DATETIME_FORMAT,
    get_datetime_string,
    chat_response_stream_multiturn_engine,
    ChatInterfaceDemo,
    CustomizedChatInterface,
)

from gradio.events import Events

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

from ..globals import MODEL_ENGINE

from ..configs import (
    USE_PANEL,
    IMAGE_TOKEN,
    IMAGE_TOKEN_INTERACTIVE,
    CHATBOT_HEIGHT,
)



CSS = """
.message-fit {
    min-width: 20em; 
    width: fit-content !important;
}

.message.svelte-1lcyrx4.svelte-1lcyrx4.svelte-1lcyrx4 {
    padding-top: 1em;
    padding-bottom: 1em;
}
"""


DOC_TEMPLATE = """###
{content}
###

"""

DOC_INSTRUCTION = """Answer the following query exclusively based on the information provided in the document above. \
If the information is not found, please say so instead of making up facts! Remember to answer the question in the same language as the user query!
"""


def undo_history(history):
    if len(history) == 0:
        return history
    if history[-1][-1] is not None:
        if history[-1][0] is not None:
            history[-1][-1] = None
        else:
            history = history[:-1]
    else:
        history = history[:-1]
    return history


def undo_history_until_last_assistant_turn(history):
    history = undo_history(history)
    while len(history) > 0 and history[-1][-1] is None:
        history = undo_history(history)
    return history, history


class MultiModalChatInterface(CustomizedChatInterface):
    def __init__(
        self,
        fn: Callable,
        *,
        chatbot: Chatbot | None = None,
        textbox: Textbox | None = None,
        additional_inputs: str | Component | list[str | Component] | None = None,
        additional_inputs_accordion_name: str | None = None,
        additional_inputs_accordion: str | Accordion | None = None,
        add_multimodal_fn: Callable | None = None,
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
        submit_btn: str | None | Button = "Submit",
        stop_btn: str | None | Button = "Stop",
        retry_btn: str | None | Button = "🔄  Retry",
        undo_btn: str | None | Button = "↩️ Undo",
        clear_btn: str | None | Button = "🗑️  Clear",
        autofocus: bool = True,
        concurrency_limit: int | None | Literal["default"] = "default",
        fill_height: bool = True,
    ):
        """
        Parameters:
            fn: The function to wrap the chat interface around. Should accept two parameters: a string input message and list of two-element lists of the form [[user_message, bot_message], ...] representing the chat history, and return a string response. See the Chatbot documentation for more information on the chat history format.
            chatbot: An instance of the gr.Chatbot component to use for the chat interface, if you would like to customize the chatbot properties. If not provided, a default gr.Chatbot component will be created.
            textbox: An instance of the gr.Textbox component to use for the chat interface, if you would like to customize the textbox properties. If not provided, a default gr.Textbox component will be created.
            additional_inputs: An instance or list of instances of gradio components (or their string shortcuts) to use as additional inputs to the chatbot. If components are not already rendered in a surrounding Blocks, then the components will be displayed under the chatbot, in an accordion.
            additional_inputs_accordion_name: Deprecated. Will be removed in a future version of Gradio. Use the `additional_inputs_accordion` parameter instead.
            additional_inputs_accordion: If a string is provided, this is the label of the `gr.Accordion` to use to contain additional inputs. A `gr.Accordion` object can be provided as well to configure other properties of the container holding the additional inputs. Defaults to a `gr.Accordion(label="Additional Inputs", open=False)`. This parameter is only used if `additional_inputs` is provided.
            examples: Sample inputs for the function; if provided, appear below the chatbot and can be clicked to populate the chatbot input.
            cache_examples: If True, caches examples in the server for fast runtime in examples. The default option in HuggingFace Spaces is True. The default option elsewhere is False.
            title: a title for the interface; if provided, appears above chatbot in large font. Also used as the tab title when opened in a browser window.
            description: a description for the interface; if provided, appears above the chatbot and beneath the title in regular font. Accepts Markdown and HTML content.
            theme: Theme to use, loaded from gradio.themes.
            css: Custom css as a string or path to a css file. This css will be included in the demo webpage.
            js: Custom js or path to js file to run when demo is first loaded. This javascript will be included in the demo webpage.
            head: Custom html to insert into the head of the demo webpage. This can be used to add custom meta tags, scripts, stylesheets, etc. to the page.
            analytics_enabled: Whether to allow basic telemetry. If None, will use GRADIO_ANALYTICS_ENABLED environment variable if defined, or default to True.
            submit_btn: Text to display on the submit button. If None, no button will be displayed. If a Button object, that button will be used.
            stop_btn: Text to display on the stop button, which replaces the submit_btn when the submit_btn or retry_btn is clicked and response is streaming. Clicking on the stop_btn will halt the chatbot response. If set to None, stop button functionality does not appear in the chatbot. If a Button object, that button will be used as the stop button.
            retry_btn: Text to display on the retry button. If None, no button will be displayed. If a Button object, that button will be used.
            undo_btn: Text to display on the delete last button. If None, no button will be displayed. If a Button object, that button will be used.
            clear_btn: Text to display on the clear button. If None, no button will be displayed. If a Button object, that button will be used.
            autofocus: If True, autofocuses to the textbox when the page loads.
            concurrency_limit: If set, this is the maximum number of chatbot submissions that can be running simultaneously. Can be set to None to mean no limit (any number of chatbot submissions can be running simultaneously). Set to "default" to use the default concurrency limit (defined by the `default_concurrency_limit` parameter in `.queue()`, which is 1 by default).
            fill_height: If True, the chat interface will expand to the height of window.
        """
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
            # Handle old gradio versions without fill_height
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
        self.add_multimodal_fn = add_multimodal_fn
        self.render_additional_inputs_fn = render_additional_inputs_fn
        self.multimodal_inputs = []
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
                            show_label=False,
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


            any_unrendered_inputs = any(
                not inp.is_rendered for inp in self.additional_inputs
            )
            if self.add_multimodal_fn is not None:
                with Row():
                    self.multimodal_inputs = self.add_multimodal_fn()
                    if self.additional_inputs and any_unrendered_inputs:
                        with Accordion(**self.additional_inputs_accordion_params):  # type: ignore
                            if self.render_additional_inputs_fn is not None:
                                self.render_additional_inputs_fn()
                            else:
                                for input_component in self.additional_inputs:
                                    if not input_component.is_rendered:
                                        input_component.render()
            else:
                if self.additional_inputs and any_unrendered_inputs:
                    with Accordion(**self.additional_inputs_accordion_params):  # type: ignore
                        if self.render_additional_inputs_fn is not None:
                            self.render_additional_inputs_fn()
                        else:
                            for input_component in self.additional_inputs:
                                if not input_component.is_rendered:
                                    input_component.render()

            if examples:
                if self.is_generator:
                    examples_fn = self._examples_stream_fn
                else:
                    # examples_fn = self._examples_fn
                    raise NotImplementedError(f'Not streaming not impl')

                self.examples_handler = Examples(
                    examples=examples,
                    inputs=[self.textbox] + self.multimodal_inputs + self.additional_inputs,
                    outputs=self.chatbot,
                    fn=examples_fn,
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
        
    def _clear_and_save_textbox(self, message: str, *multimodal_inputs) -> tuple[str, str]:
        saved_input = [message] + list(multimodal_inputs)
        outputs = [''] + [None] * len(multimodal_inputs)
        return outputs + [saved_input]
    
    def _add_inputs_to_history(self, history: List[List[Union[str, None]]], *args):
        message = args[0]
        multimodal_inputs = args[1:1 + len(self.multimodal_inputs)] if len(args) > 1 else None
        if multimodal_inputs is not None:
            is_file_exists = [(x is not None and os.path.exists(x)) for x in multimodal_inputs]
            if any(is_file_exists):
                file_exists = [f for f, ise in zip(multimodal_inputs, is_file_exists) if ise]
                if len(file_exists) > 1:
                    raise gr.Error(f"Cannot have more than 1 multimodal input at a time.")
                fname = file_exists[0]
                history.append([(fname,), None])
        if message is not None and message.strip() != "":
            history.append([message, None])
        return history


    def _display_input(
        self, saved_input: List[str], history: List[List[Union[str, None]]]
    ) -> Tuple[List[List[Union[str, None]]], List[List[list[Union[str, None]]]]]:
        # message = saved_input[0]
        # multimodal_inputs = saved_input[1:] if len(saved_input) > 1 else None
        # # ! If things wrong, return original history and give warning
        # if multimodal_inputs is not None:
        #     is_file_exists = [(x is not None and os.path.exists(x)) for x in multimodal_inputs]
        #     if any(is_file_exists):
        #         file_exists = [f for f, ise in zip(multimodal_inputs, is_file_exists) if ise]
        #         if len(file_exists) > 1:
        #             raise gr.Error(f"Cannot have more than 1 multimodal input at a time.")
        #         fname = file_exists[0]
        #         history.append([(fname,), None])
        # if message is not None and message.strip() != "":
        #     history.append([message, None])
        history = self._add_inputs_to_history(history, *saved_input)
        return history, history
    
    def _delete_prev_fn(
        self, history: list[list[str | None]]
    ) -> tuple[list[list[str | None]], str, list[list[str | None]]]:
        try:
            message, _ = history.pop()
        except IndexError:
            message = ""
        saved_input = [message or ""] + [None] * len(self.multimodal_inputs)
        return history, saved_input, history
    
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
                    [self.textbox] + self.multimodal_inputs,
                    [self.textbox] + self.multimodal_inputs + [self.saved_input],
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
                .success(
                    submit_fn,
                    [self.chatbot_state] + self.additional_inputs,
                    [self.chatbot, self.chatbot_state, self.num_tokens],
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
                .success(
                    submit_fn,
                    [self.chatbot_state] + self.additional_inputs,
                    [self.chatbot, self.chatbot_state, self.num_tokens],
                    api_name=False,
                )
            )
            self._setup_stop_events([self.retry_btn.click], retry_event)

        if self.undo_btn:
            self.undo_btn.click(
                # self._delete_prev_fn,
                # [self.chatbot_state],
                # [self.chatbot, self.saved_input, self.chatbot_state],
                undo_history_until_last_assistant_turn,
                [self.chatbot_state],
                [self.chatbot, self.chatbot_state],
                api_name=False,
                queue=False,
            )
            # .then(
            #     lambda x: x,
            #     [self.saved_input],
            #     [self.textbox],
            #     api_name=False,
            #     queue=False,
            # )

    async def _stream_fn(
        self,
        # message: str,
        history_with_input,
        request: Request,
        *args,
    ) -> AsyncGenerator:
        history = history_with_input[:-1]
        message = history_with_input[-1][0]
        inputs, _, _ = special_args(
            self.fn, inputs=[history_with_input, *args], request=request
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
                first_response, num_tokens = first_response_pack
            else:
                first_response, num_tokens = first_response_pack, -1
            update = history + [[message, first_response]]
            yield update, update, f"{num_tokens} toks"
        except StopIteration:
            update = history + [[message, None]]
            yield update, update, "NaN toks"
        except Exception as e:
            yield history, history, "NaN toks"
            raise e

        try:
            async for response_pack in generator:
                if isinstance(response_pack, (tuple, list)):
                    response, num_tokens = response_pack
                else:
                    response, num_tokens = response_pack, "NaN toks"
                update = history + [[message, response]]
                yield update, update, f"{num_tokens} toks"
        except Exception as e:
            yield history, history, "NaN toks"
            raise e
    
    async def _examples_stream_fn(
        self,
        # message: str,
        *args,
    ) -> AsyncGenerator:
        history = []
        input_len = 1 + len(self.multimodal_inputs)
        saved_input = args[:input_len]
        message = saved_input[0]
        additional_inputs = [] if len(args) <= input_len else args[input_len:]
        history = self._add_inputs_to_history(history, *saved_input)
        inputs, _, _ = special_args(self.fn, inputs=[history, *additional_inputs], request=None)

        if self.is_async:
            generator = self.fn(*inputs)
        else:
            generator = await anyio.to_thread.run_sync(
                self.fn, *inputs, limiter=self.limiter
            )
            generator = SyncToAsyncIterator(generator, self.limiter)
        # async for response in generator:
        #     yield [[message, response]]
        
        try:
            async for response_pack in generator:
                if isinstance(response_pack, (tuple, list)):
                    response, num_tokens = response_pack
                else:
                    response, num_tokens = response_pack, "NaN toks"
                update = history + [[message, response]]
                yield update, update, f"{num_tokens} toks"
        except Exception as e:
            yield history, history, "NaN toks"
            raise e
    
    async def _examples_fn(self, message: str, *args) -> list[list[str | None]]:
        raise NotImplementedError
        inputs, _, _ = special_args(self.fn, inputs=[message, [], *args], request=None)

        if self.is_async:
            response = await self.fn(*inputs)
        else:
            response = await anyio.to_thread.run_sync(
                self.fn, *inputs, limiter=self.limiter
            )
        return [[message, response]]



def gradio_history_to_openai_conversations(message=None, history=None, system_prompt=None):
    conversations = []
    system_prompt = system_prompt or SYSTEM_PROMPT
    if history is not None and len(history) > 0:
        for i, (prompt, res) in enumerate(history):
            if prompt is not None:
                conversations.append({"role": "user", "content": prompt.strip()})
            if res is not None:
                conversations.append({"role": "assistant", "content": res.strip()})
    if message is not None:
        if len(message.strip()) == 0:
            raise gr.Error("The message cannot be empty!")
        conversations.append({"role": "user", "content": message.strip()})
    if conversations[0]['role'] != 'system':
        conversations = [{"role": "system", "content": system_prompt}] + conversations
    return conversations


def gradio_history_to_conversation_prompt(message=None, history=None, system_prompt=None):
    global MODEL_ENGINE
    full_prompt = MODEL_ENGINE.apply_chat_template(
        gradio_history_to_openai_conversations(
            message, history=history, system_prompt=system_prompt),
        add_generation_prompt=True
    )
    return full_prompt


def gradio_history_to_vision_conversation_prompt_paths(
        history, system_prompt=None, image_token=None
):
    """
    Aggregate gradio history into openai conversations
    history = [
        ["Hello", "Response"],
        [(file,), None],
    ]
    --->
    [
        {"role": "user", "content": ...}
    ]
    """
    global MODEL_ENGINE
    image_token = image_token or IMAGE_TOKEN
    conversations = []
    image_paths = []
    for i, his in enumerate(history):
        prompt, response = his
        last_turn = conversations[-1] if len(conversations) > 0 else None
        if prompt is not None:
            if isinstance(prompt, tuple):
                image_path = prompt[0]
                if last_turn is not None and last_turn['role'] == 'user':
                    last_turn['content'] += f" {image_token}"
                else:
                    # last_turn None or last_turn['role'] == 'assistant'
                    conversations.append({
                        "role": "user",
                        "content": f"{image_token}"
                    })
                image_paths.append(image_path)
            else:
                assert prompt is not None and isinstance(prompt, str)
                if last_turn is not None and last_turn['role'] == 'user':
                    last_turn['content'] += f"\n{prompt}"
                else:
                    conversations.append({
                        "role": "user",
                        "content": prompt,
                    })
        if response is not None:
            assert isinstance(response, str)
            conversations.append({
                "role": "assistant",
                "content": response,
            })

    if conversations[0]['role'] != 'system':
        system_prompt = system_prompt or SYSTEM_PROMPT
        conversations = [{"role": "system", "content": system_prompt}] + conversations
    
    # print(f'convo: {json.dumps(conversations, indent=4, ensure_ascii=False)}\n{image_paths=}')
    full_prompt = MODEL_ENGINE.apply_chat_template(
        conversations,
        add_generation_prompt=True
    )
    return full_prompt, image_paths, conversations


def is_doc(file_path):
    is_doc_allowed = file_path.endswith((".pdf", ".docx", ".txt"))
    return is_doc_allowed


def read_doc(file_path):
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    texts = loader.load()
    text = "\n\n".join([t.page_content for t in texts])
    return text


def doc_file_to_instruct_content(file_path, doc_instruction=None):
    doc_instruction = doc_instruction or DOC_INSTRUCTION
    content = doc_instruction.strip() + "\n" + DOC_TEMPLATE.format(content=read_doc(file_path))
    return content


def gradio_history_to_doc_conversation_prompt(
        history, system_prompt=None, doc_instruction=None,
):
    """
    Aggregate gradio history into openai conversations
    history = [
        ["Hello", "Response"],
        [(file,), None],
    ]
    --->
    [
        {"role": "user", "content": ...}
    ]
    """
    global MODEL_ENGINE
    # image_token = image_token or IMAGE_TOKEN
    doc_instruction = doc_instruction or DOC_INSTRUCTION
    conversations = []
    image_paths = []
    for i, his in enumerate(history):
        prompt, response = his
        last_turn = conversations[-1] if len(conversations) > 0 else None
        if prompt is not None:
            if isinstance(prompt, tuple):
                file_path = prompt[0]
                if not is_doc(file_path):
                    raise gr.Error(f'file not doc {file_path}')
                content = doc_file_to_instruct_content(file_path, doc_instruction)
                if last_turn is not None and last_turn['role'] == 'user':
                    last_turn['content'] += f"{content}"
                else:
                    # last_turn None or last_turn['role'] == 'assistant'
                    conversations.append({
                        "role": "user",
                        "content": f"{content}"
                    })
            else:
                assert prompt is not None and isinstance(prompt, str)
                if last_turn is not None and last_turn['role'] == 'user':
                    last_turn['content'] += f"\n{prompt}"
                else:
                    conversations.append({
                        "role": "user",
                        "content": prompt,
                    })
        if response is not None:
            assert isinstance(response, str)
            conversations.append({
                "role": "assistant",
                "content": response,
            })

    if conversations[0]['role'] != 'system':
        system_prompt = system_prompt or SYSTEM_PROMPT
        conversations = [{"role": "system", "content": system_prompt}] + conversations
    
    full_prompt = MODEL_ENGINE.apply_chat_template(
        conversations,
        add_generation_prompt=True
    )
    return full_prompt, conversations


def gradio_history_to_vision_doc_conversation_prompt_paths(
        history, system_prompt=None, image_token=None, doc_instruction=None,
):
    """
    Aggregate gradio history into openai conversations
    history = [
        ["Hello", "Response"],
        [(file,), None],
    ]
    --->
    [
        {"role": "user", "content": ...}
    ]
    """
    global MODEL_ENGINE
    image_token = image_token or IMAGE_TOKEN
    doc_instruction = doc_instruction or DOC_INSTRUCTION
    conversations = []
    image_paths = []
    for i, his in enumerate(history):
        prompt, response = his
        last_turn = conversations[-1] if len(conversations) > 0 else None
        if prompt is not None:
            if isinstance(prompt, tuple):
                file_path = prompt[0]
                if is_doc(file_path):
                    content = doc_file_to_instruct_content(file_path, doc_instruction)
                    if last_turn is not None and last_turn['role'] == 'user':
                        last_turn['content'] += f"{content}"
                    else:
                        # last_turn None or last_turn['role'] == 'assistant'
                        conversations.append({
                            "role": "user",
                            "content": f"{content}"
                        })
                else:
                    if last_turn is not None and last_turn['role'] == 'user':
                        last_turn['content'] += f" {image_token}"
                    else:
                        # last_turn None or last_turn['role'] == 'assistant'
                        conversations.append({
                            "role": "user",
                            "content": f"{image_token}"
                        })
                    image_paths.append(file_path)
            else:
                assert prompt is not None and isinstance(prompt, str)
                if last_turn is not None and last_turn['role'] == 'user':
                    last_turn['content'] += f"\n{prompt}"
                else:
                    conversations.append({
                        "role": "user",
                        "content": prompt,
                    })
        if response is not None:
            assert isinstance(response, str)
            conversations.append({
                "role": "assistant",
                "content": response,
            })

    if conversations[0]['role'] != 'system':
        system_prompt = system_prompt or SYSTEM_PROMPT
        conversations = [{"role": "system", "content": system_prompt}] + conversations
    
    full_prompt = MODEL_ENGINE.apply_chat_template(
        conversations,
        add_generation_prompt=True
    )
    return full_prompt, image_paths, conversations


def vision_chat_response_stream_multiturn_engine(
    history: List[Tuple[str, str]], 
    temperature: float, 
    max_tokens: int, 
    system_prompt: Optional[str] = SYSTEM_PROMPT,
    image_token: Optional[str] = IMAGE_TOKEN,
):
    global MODEL_ENGINE
    temperature = float(temperature)
    # ! remove frequency_penalty
    # frequency_penalty = float(frequency_penalty)
    max_tokens = int(max_tokens)
    # ! skip safety
    if DATETIME_FORMAT in system_prompt:
        # ! This sometime works sometimes dont
        system_prompt = system_prompt.format(cur_datetime=get_datetime_string())
    # ! history now can have multimodal
        
    full_prompt, image_paths, conversations = gradio_history_to_vision_conversation_prompt_paths(
        history=history, system_prompt=system_prompt, image_token=image_token
    )

    if hasattr(MODEL_ENGINE, "get_multimodal_tokens"):
        num_tokens = MODEL_ENGINE.get_multimodal_tokens(full_prompt, image_paths=image_paths)
    else:
        num_tokens = len(MODEL_ENGINE.tokenizer.encode(full_prompt))
    if num_tokens >= MODEL_ENGINE.max_position_embeddings - 128:
        raise gr.Error(f"Conversation or prompt is too long ({num_tokens} toks), please clear the chatbox or try shorter input.")
    
    print(f'{image_paths=}')
    print(full_prompt)
    outputs = None
    response = None
    num_tokens = -1
    for j, outputs in enumerate(MODEL_ENGINE.generate_yield_string(
        prompt=full_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        image_paths=image_paths,
    )):
        if isinstance(outputs, tuple):
            response, num_tokens = outputs
        else:
            response, num_tokens = outputs, -1
        yield response, num_tokens
    
    print(format_conversation(history + [[None, response]]))
    
    if response is not None:
        yield response, num_tokens


def doc_chat_response_stream_multiturn_engine(
    history: List[Tuple[str, str]], 
    temperature: float, 
    max_tokens: int, 
    system_prompt: Optional[str] = SYSTEM_PROMPT,
    doc_instruction: Optional[str] = DOC_INSTRUCTION,
):
    global MODEL_ENGINE
    temperature = float(temperature)
    # ! remove frequency_penalty
    # frequency_penalty = float(frequency_penalty)
    max_tokens = int(max_tokens)
    # ! skip safety
    if DATETIME_FORMAT in system_prompt:
        # ! This sometime works sometimes dont
        system_prompt = system_prompt.format(cur_datetime=get_datetime_string())
    # ! history now can have multimodal
        
    full_prompt, conversations = gradio_history_to_doc_conversation_prompt(
        history=history, system_prompt=system_prompt, doc_instruction=doc_instruction
    )

    # ! length checked
    num_tokens = len(MODEL_ENGINE.tokenizer.encode(full_prompt))
    if num_tokens >= MODEL_ENGINE.max_position_embeddings - 128:
        raise gr.Error(f"Conversation or prompt is too long ({num_tokens} toks), please clear the chatbox or try shorter input.")
    
    print(full_prompt)
    outputs = None
    response = None
    num_tokens = -1
    for j, outputs in enumerate(MODEL_ENGINE.generate_yield_string(
        prompt=full_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        # image_paths=image_paths,
    )):
        if isinstance(outputs, tuple):
            response, num_tokens = outputs
        else:
            response, num_tokens = outputs, -1
        yield response, num_tokens
    
    print(format_conversation(history + [[None, response]]))
    
    if response is not None:
        yield response, num_tokens




def vision_doc_chat_response_stream_multiturn_engine(
    history: List[Tuple[str, str]], 
    temperature: float, 
    max_tokens: int, 
    system_prompt: Optional[str] = SYSTEM_PROMPT,
    image_token: Optional[str] = IMAGE_TOKEN,
    doc_instruction: Optional[str] = DOC_INSTRUCTION,
):
    global MODEL_ENGINE
    temperature = float(temperature)
    # ! remove frequency_penalty
    # frequency_penalty = float(frequency_penalty)
    max_tokens = int(max_tokens)
    # ! skip safety
    if DATETIME_FORMAT in system_prompt:
        # ! This sometime works sometimes dont
        system_prompt = system_prompt.format(cur_datetime=get_datetime_string())
    # ! history now can have multimodal
        
    full_prompt, image_paths, conversations = gradio_history_to_vision_doc_conversation_prompt_paths(
        history=history, system_prompt=system_prompt, image_token=image_token, doc_instruction=doc_instruction
    )

    # ! length check
    if hasattr(MODEL_ENGINE, "get_multimodal_tokens"):
        num_tokens = MODEL_ENGINE.get_multimodal_tokens(full_prompt, image_paths=image_paths)
    else:
        num_tokens = len(MODEL_ENGINE.tokenizer.encode(full_prompt))
    if num_tokens >= MODEL_ENGINE.max_position_embeddings - 128:
        raise gr.Error(f"Conversation or prompt is too long ({num_tokens} toks), please clear the chatbox or try shorter input.")
    
    print(full_prompt)
    print(f'{image_paths=}')
    outputs = None
    response = None
    num_tokens = -1
    for j, outputs in enumerate(MODEL_ENGINE.generate_yield_string(
        prompt=full_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        image_paths=image_paths,
    )):
        if isinstance(outputs, tuple):
            response, num_tokens = outputs
        else:
            response, num_tokens = outputs, -1
        yield response, num_tokens
    
    print(format_conversation(history + [[None, response]]))
    
    if response is not None:
        yield response, num_tokens



@register_demo
class VisionChatInterfaceDemo(ChatInterfaceDemo):
    """
    Accept vision image
    """

    @property
    def tab_name(self):
        return "Vision Chat"
    
    @property
    def examples(self):
        return [
            ["What's strange about this image?", "assets/dog_monalisa.jpeg",],
            ["Explain why the sky is blue.", None,],
        ]

    def create_demo(
            self, 
            title: str | None = None, 
            description: str | None = None, 
            **kwargs
        ) -> gr.Blocks:
        system_prompt = kwargs.get("system_prompt", SYSTEM_PROMPT)
        max_tokens = kwargs.get("max_tokens", MAX_TOKENS)
        temperature = kwargs.get("temperature", TEMPERATURE)
        model_name = kwargs.get("model_name", MODEL_NAME)
        description = description or """Upload an image to ask question about it."""

        def add_multimodal_fn() -> List[Component]:
            image_input = gr.Image(label="Input Image", type="filepath", )
            return [image_input]

        additional_inputs = [
            gr.Number(value=temperature, label='Temperature', min_width=20), 
            gr.Number(value=max_tokens, label='Max-tokens', min_width=20), 
            gr.Textbox(value=system_prompt, label='System prompt', lines=1),
            gr.Textbox(value=IMAGE_TOKEN, label='Visual token', lines=1, interactive=IMAGE_TOKEN_INTERACTIVE, min_width=20),
        ]
        def render_additional_inputs_fn():
            with Row():
                additional_inputs[0].render()
                additional_inputs[1].render()
                additional_inputs[3].render()
            additional_inputs[2].render()

        demo_chat = MultiModalChatInterface(
            vision_chat_response_stream_multiturn_engine,
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
            # textbox=gr.Textbox(placeholder='Type message', lines=4, max_lines=128, min_width=200),
            textbox=gr.Textbox(placeholder='Type message', lines=1, max_lines=128, min_width=200, scale=8),
            submit_btn=gr.Button(value='Submit', variant="primary", scale=0),
            # ! consider preventing the stop button
            # stop_btn=None,
            add_multimodal_fn=add_multimodal_fn,
            title=title,
            description=description,
            additional_inputs=additional_inputs, 
            render_additional_inputs_fn=render_additional_inputs_fn,
            additional_inputs_accordion=gr.Accordion("Additional Inputs", open=True),
            examples=self.examples,
            cache_examples=False,
            css=CSS,
        )
        return demo_chat


def add_document_upload():
    file_input = gr.File(label='Upload pdf, docx, txt', file_count='single', file_types=['pdf', 'docx', 'txt'])
    # ! Some platform has problems with gr.File, so use uploadbutton instead
    # with Group():
    #     file_input = gr.Textbox(value=None, label='Document path', lines=1, interactive=False)
    #     upload_button = gr.UploadButton("Click to Upload document", file_types=['pdf', 'docx', 'txt'], file_count="single")
    #     upload_button.upload(lambda x: x.name, upload_button, file_input)
    return file_input


@register_demo
class DocChatInterfaceDemo(ChatInterfaceDemo):
    """
    Accept document (full length no RAG)
    """
    @property
    def tab_name(self):
        return "Doc Chat"
    
    @property
    def examples(self):
        return [
            ["Summarize the document", "assets/attention_short.pdf",],
            ["Explain why the sky is blue.", None,],
        ]
    
    def create_demo(
            self, 
            title: str | None = None, 
            description: str | None = None, 
            **kwargs
        ) -> gr.Blocks:
        system_prompt = kwargs.get("system_prompt", SYSTEM_PROMPT)
        max_tokens = kwargs.get("max_tokens", MAX_TOKENS)
        temperature = kwargs.get("temperature", TEMPERATURE)
        model_name = kwargs.get("model_name", MODEL_NAME)
        # frequence_penalty = FREQUENCE_PENALTY
        # presence_penalty = PRESENCE_PENALTY
        description = description or """Upload a short document to ask question about it."""

        def add_multimodal_fn() -> List[Component]:
            file_input = add_document_upload()
            # image_input = gr.Image(label="Input Image", type="filepath", )
            return [file_input]
        
        additional_inputs = [
            gr.Number(value=temperature, label='Temperature', min_width=20), 
            gr.Number(value=max_tokens, label='Max-tokens', min_width=20), 
            gr.Textbox(value=system_prompt, label='System prompt', lines=1),
            gr.Textbox(value=DOC_INSTRUCTION, label='Doc instruction', lines=1),
        ]
        def render_additional_inputs_fn():
            with Row():
                additional_inputs[0].render()
                additional_inputs[1].render()
            additional_inputs[2].render()
            additional_inputs[3].render()

        demo_chat = MultiModalChatInterface(
            doc_chat_response_stream_multiturn_engine,
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
            # ! consider preventing the stop button
            add_multimodal_fn=add_multimodal_fn,
            title=title,
            description=description,
            additional_inputs=additional_inputs, 
            render_additional_inputs_fn=render_additional_inputs_fn,
            additional_inputs_accordion=gr.Accordion("Additional Inputs", open=True),
            examples=self.examples,
            cache_examples=False,
            css=CSS,
        )
        return demo_chat


@register_demo
class VisionDocChatInterfaceDemo(ChatInterfaceDemo):
    """
    Accept either vision image or document (full length no RAG)
    """
    @property
    def tab_name(self):
        return "Vision Doc Chat"

    @property
    def examples(self):
        return [
            ["What's strange about this image?", None, "assets/dog_monalisa.jpeg",],
            ["Summarize the document", "assets/attention_short.pdf", None,],
            ["Explain why the sky is blue.", None, None],
        ]
    
    def create_demo(
            self, 
            title: str | None = None, 
            description: str | None = None, 
            **kwargs
        ) -> gr.Blocks:
        system_prompt = kwargs.get("system_prompt", SYSTEM_PROMPT)
        max_tokens = kwargs.get("max_tokens", MAX_TOKENS)
        temperature = kwargs.get("temperature", TEMPERATURE)
        model_name = kwargs.get("model_name", MODEL_NAME)
        # frequence_penalty = FREQUENCE_PENALTY
        # presence_penalty = PRESENCE_PENALTY
        description = description or """Upload either an image or short document to ask question about it."""

        def add_multimodal_fn() -> List[Component]:
            file_input = add_document_upload()
            image_input = gr.Image(label="Input Image", type="filepath", )
            return [file_input, image_input]

        additional_inputs = [
            gr.Number(value=temperature, label='Temperature', min_width=20), 
            gr.Number(value=max_tokens, label='Max-tokens', min_width=20), 
            gr.Textbox(value=system_prompt, label='System prompt', lines=1),
            gr.Textbox(value=IMAGE_TOKEN, label='Visual token', lines=1, interactive=IMAGE_TOKEN_INTERACTIVE, min_width=2),
            gr.Textbox(value=DOC_INSTRUCTION, label='Doc instruction', lines=1),
        ]
        def render_additional_inputs_fn():
            with Row():
                additional_inputs[0].render()
                additional_inputs[1].render()
                additional_inputs[3].render()
            additional_inputs[2].render()
            additional_inputs[4].render()

        demo_chat = MultiModalChatInterface(
            vision_doc_chat_response_stream_multiturn_engine,
            chatbot=gr.Chatbot(
                label=MODEL_NAME,
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
            add_multimodal_fn=add_multimodal_fn,
            title=title,
            description=description,
            additional_inputs=additional_inputs, 
            render_additional_inputs_fn=render_additional_inputs_fn,
            additional_inputs_accordion=gr.Accordion("Additional Inputs", open=True),
            examples=self.examples,
            cache_examples=False,
            css=CSS,
        )
        return demo_chat