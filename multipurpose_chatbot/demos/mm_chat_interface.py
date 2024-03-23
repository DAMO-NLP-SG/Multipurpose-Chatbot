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

from .multimodal_chat_interface import (
    undo_history,
    undo_history_until_last_assistant_turn,
    vision_chat_response_stream_multiturn_engine,
    doc_chat_response_stream_multiturn_engine,
    vision_doc_chat_response_stream_multiturn_engine,
    gradio_history_to_conversation_prompt,
    gradio_history_to_openai_conversations,
    gradio_history_to_doc_conversation_prompt,
    gradio_history_to_vision_conversation_prompt_paths,
    gradio_history_to_vision_doc_conversation_prompt_paths,
)

# .message-fit {
#     min-width: 20em; 
#     width: fit-content !important;
# }

EXAMPLES_PER_PAGE = int(os.environ.get("EXAMPLES_PER_PAGE", 10))

CSS = """
.message.svelte-1lcyrx4.svelte-1lcyrx4.svelte-1lcyrx4 {
    padding-top: 1em;
}
"""

CSS = """
.panel-full-width.svelte-1lcyrx4.svelte-1lcyrx4.svelte-1lcyrx4 {
    padding: calc(var(--spacing-xxl) * 1);
    width: 100%
}
"""

DOC_TEMPLATE = """###
{content}
###

"""

DOC_INSTRUCTION = """Answer the following query exclusively based on the information provided in the document above. \
If the information is not found, please say so instead of making up facts! Remember to answer the question in the same language as the user query!
"""


MultimodalTextbox = None

try:
    from gradio import MultimodalTextbox
except ImportError as e:
    print(f'Cannot import MultiMOdalTextbox: {MultimodalTextbox}')


class MultiModalTextChatInterface(CustomizedChatInterface):
    def __init__(
        self,
        fn: Callable,
        *,
        chatbot: Chatbot | None = None,
        textbox: Textbox | None = None,
        additional_inputs: str | Component | list[str | Component] | None = None,
        additional_inputs_accordion_name: str | None = None,
        additional_inputs_accordion: str | Accordion | None = None,
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
            
            # =------
            with Row():
                if textbox:
                    # textbox.container = False
                    # textbox.show_label = False
                    textbox_ = textbox.render()
                    # assert isinstance(textbox_, Textbox)
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
                self.buttons.extend([stop_btn])  # type: ignore

                self.num_tokens = Textbox(
                        # container=False,
                        show_label=False,
                        label="# Tokens",
                        placeholder="0 tokens",
                        scale=1,
                        interactive=False,
                        # autofocus=autofocus,
                        min_width=10
                    )
            
            self.fake_api_btn = Button("Fake API", visible=False)
            self.fake_response_textbox = Textbox(label="Response", visible=False)
            (
                self.retry_btn,
                self.undo_btn,
                self.clear_btn,
                # self.submit_btn,
                self.stop_btn,
            ) = self.buttons
            self.submit_btn = None

            if examples:
                if self.is_generator:
                    examples_fn = self._examples_stream_fn
                else:
                    # examples_fn = self._examples_fn
                    raise NotImplementedError()
                
                def copy_to_mm_textbox(message, image, filename):
                    save_input = {"text": message, "files": []}
                    if filename is not None and os.path.exists(filename):
                        # save_input['files'].append({"path": file})
                        save_input['files'].append(filename)
                    if image is not None and os.path.exists(image):
                        # save_input['files'].append({"path": file})
                        save_input['files'].append(image)
                    print(save_input)
                    return save_input
                
                # self.example_textbox = gr.Textbox(visible=False)
                # self.example_file = gr.File(file_count='single', type='filepath', visible=False)
                # self.example_image = gr.Image(type='filepath', visible=False)

                # self.examples_handler = Examples(
                #     examples=examples,
                #     inputs=[self.example_textbox, self.example_image, self.example_file],
                #     outputs=self.textbox,
                #     # fn=examples_fn,
                #     fn=copy_to_mm_textbox,
                #     run_on_click=True
                # )
                self.examples_handler = Examples(
                    examples=examples,
                    # inputs=[self.textbox] + self.additional_inputs,
                    inputs=[self.textbox],
                    # outputs=self.chatbot,
                    # fn=examples_fn,
                    examples_per_page=EXAMPLES_PER_PAGE,
                )

            any_unrendered_inputs = any(
                not inp.is_rendered for inp in self.additional_inputs
            )
            if self.additional_inputs and any_unrendered_inputs:
                with Accordion(**self.additional_inputs_accordion_params):  # type: ignore
                    for input_component in self.additional_inputs:
                        if not input_component.is_rendered:
                            input_component.render()

            # The example caching must happen after the input components have rendered
            if cache_examples:
                client_utils.synchronize_async(self.examples_handler.cache)

            self.saved_input = State()
            self.chatbot_state = (
                State(self.chatbot.value) if self.chatbot.value else State([])
            )

            self._setup_events()
            self._setup_api()

    def _clear_and_save_textbox(self, saved_input: Dict[str, Union[str, list]]) -> Tuple[Dict[str, Union[str, list]], Dict[str, Union[str, list]]]:
        return {"text": "", "files": []}, saved_input

    def _add_inputs_to_history(self, history: List[List[Union[str, None]]], save_input: Dict[str, Union[str, list]]):
        message = save_input['text']
        files = save_input['files']
        if files is not None and len(files) > 0:
            for f in files:
                fpath = f['path'] if isinstance(f, dict) else f
                history.append([(fpath, ), None])
        if message is not None and message.strip() != "":
            history.append([message, None])
        return history
    
    def _display_input(
        self, saved_input: Dict[str, Union[str, list]], history: List[List[Union[str, None]]]
    ) -> Tuple[List[List[Union[str, None]]], List[List[list[Union[str, None]]]]]:
        message = saved_input["text"]
        files = saved_input['files']
        if files is not None and len(files) > 0:
            print(files)
            for f in files:
                fpath = f['path'] if isinstance(f, dict) else f
                history.append([(fpath, ), None])
        if message is not None and message.strip() != "":
            history.append([message, None])
        return history, history

    def _delete_prev_fn(
        self, history: list[list[str | None]]
    ) -> tuple[list[list[str | None]], str, list[list[str | None]]]:
        try:
            message, _ = history.pop()
        except IndexError:
            message = ""
        # saved_input = [message or ""] + [None] * len(self.multimodal_inputs)
        saved_input = {"text": message, "files": []}
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
                # [self.textbox.submit, self.submit_btn.click]
                [self.textbox.submit]
                if self.submit_btn
                else [self.textbox.submit]
            )
            submit_event = (
                on(
                    submit_triggers,
                    self._clear_and_save_textbox,
                    [self.textbox],
                    [self.textbox] + [self.saved_input],
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
            if self.submit_btn:
                self.clear_btn.click(
                    lambda: ([], [], None, Button(interactive=True)),
                    None,
                    [self.chatbot, self.chatbot_state, self.saved_input, self.submit_btn],
                    queue=False,
                    api_name=False,
                    cancels=event_to_cancel,
                )
            else:
                self.clear_btn.click(
                    lambda: ([], [], None),
                    None,
                    [self.chatbot, self.chatbot_state, self.saved_input],
                    queue=False,
                    api_name=False,
                    cancels=event_to_cancel,
                )
    
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
            # print(f"===\n{update}")
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
                # print(f"------\n{update}")
                yield update, update, f"{num_tokens} toks"
        except Exception as e:
            yield history, history, "NaN toks"
            raise e
    
    async def _examples_stream_fn(
        self,
        # message: str,
        *args,
    ) -> AsyncGenerator:
        raise ValueError(f'invalid')
        history = []
        # input_len = 1 + len(self.multimodal_inputs)
        # input_len = 2
        # saved_input = args[:input_len]
        # saved_input = args[0]
        # message = saved_input['text']
        # files = saved_input['files']
        message = args[0]
        fname = args[1]
        saved_input = {
            "text": message,
            "files": []
        }
        if fname is not None and os.path.exists(fname):
            # saved_input['files'].append({"path": fname})
            saved_input['files'].append(fname)

        additional_inputs = args[2:]
        history = self._add_inputs_to_history(history, saved_input)
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



@register_demo
class VisionMMChatInterfaceDemo(ChatInterfaceDemo):
    """
    Accept vision image
    """

    @property
    def tab_name(self):
        return "Vision Chat"
    
    @property
    def examples(self):
        from pathlib import Path
        from gradio.data_classes import FileData, GradioModel
        # return [
        #     ["What's strange about this image?", "assets/dog_monalisa.jpeg", None],
        #     ["Explain why the sky is blue.", None,],
        # ]
        return [
            # [{"text": "Summarize the document", "files": [{
            #     "path": "assets/attention_short.pdf", "orig_name": "attention_short", "mime_type": "application/pdf",
            #     "size": Path("assets/attention_short.pdf").stat().st_size
            #     }
            # ]}],
            # [{"text": "Summarize the document", "files": ["assets/attention_short.pdf"]}],
            # [{"text": "Summarize the document", "files": [
            #     FileData(
            #         path="assets/attention_short.pdf",
            #         mime_type="application/pdf",
            #         orig_name="attention_short",
            #         size=Path("assets/attention_short.pdf").stat().st_size,
            #         url="attention_short.pdf",
            #     )    
            # ]}],
            # [{"text": "What's strange about this image?", "files": ["assets/dog_monalisa.jpeg"]},],
            # [{"text": "Explain why the sky is blue.", "files": []},],
            [{"text": "Mô tả chi tiết bức ảnh.", "files": ["assets/imgs/athlete.jpeg", ]} ],
            [{"text": "Mô tả chi tiết bức ảnh.", "files": ["assets/imgs/chart_algo.png", ]} ],
            [{"text": "Explain the image.", "files": ["assets/imgs/chart_soap_sense_cycle.png", ]} ],
            [{"text": "Provide a detailed description of the poster.", "files": ["assets/imgs/covid.jpeg", ]} ],
            [{"text": "Where is this place exactly?", "files": ["assets/imgs/danang.jpeg", ]} ],
            [{"text": "What's strange about this image?", "files": ["assets/dog_monalisa.jpeg",]} ],
            [{"text": "Đây là ở đâu?", "files": ["assets/imgs/great_wall.png", ]} ],
            [{"text": "Giới thiệu về nơi này.", "files": ["assets/imgs/hochiminh_city.jpeg", ]} ],
            [{"text": "Đây là ở đâu?", "files": ["assets/imgs/hochiminh_mausoleum.jpeg", ]} ],
            [{"text": "Suy nghĩ từng bước một để tìm x.", "files": ["assets/imgs/find_x_triangle.jpeg", ]} ],
            [{"text": "Provide a detailed description of the poster.", "files": ["assets/imgs/home_injury.jpeg", ]} ],
            [{"text": "Đây là hành tinh gì?", "files": ["assets/imgs/jupyter.jpeg", ]} ],
            [{"text": "Miêu tả bức ảnh trên.", "files": ["assets/imgs/leaf.png", ]} ],
            [{"text": "Đây là đâu?", "files": ["assets/imgs/mbs.png", ]} ],
            [{"text": "Introduce this figure.", "files": ["assets/imgs/merlion_2.jpeg", ]} ],
            [{"text": "Explain the figure.", "files": ["assets/imgs/photosynthesis.png", ]} ],
            [{"text": "List out all the details of the image.", "files": ["assets/imgs/sewing_tools.png", ]} ],
            [{"text": "What happened in this photo.", "files": ["assets/imgs/tiananmen_tankman.jpeg", ]} ],
            [{"text": "Có gì ngoài 2 con mèo?", "files": ["assets/imgs/two_cats.jpeg", ]} ],
            [{"text": "Biển báo nói gì?", "files": ["assets/imgs/cau_oo.jpeg", ]} ],
            [{"text": "Đây là món gì và hướng dẫn cách làm.", "files": ["assets/imgs/banhmy.jpeg", ]} ],
            [{"text": "Hãy hướng dẫn nấu món này.", "files": ["assets/imgs/cach-nau-pho-bo-nam-dinh.jpeg", ]} ],
            [{"text": "Bức tường nói gì?", "files": ["assets/imgs/camdaibay.jpeg", ]} ],
            [{"text": "Công thức này là gì", "files": ["assets/imgs/eistein_field_equation.png", ]} ],
            [{"text": "What is this formula about?", "files": ["assets/imgs/eistein_field_equation.png", ]} ],
            [{"text": "Hãy tìm góc còn lại.", "files": ["assets/imgs/triangle_find_angle.png", ]} ],
            [{"text": "Đây là đâu?", "files": ["assets/imgs/seattle_space_needle.jpeg", ]} ],
            [{"text": "Describe the image", "files": ["assets/imgs/seal_logo.png", ]} ],
            # [{"text": "Explain why the sky is blue.", None,} ],
            [{"text": "Hãy giải thích thuyết tương đối rộng.", "files": []},],
            [{"text": "Hãy giải thích vấn đề P vs NP.", "files": []},],
            [{"text": "Explain general relativity.", "files": []},],
            [{"text": 'Vừa gà vừa chó, bó lại cho tròn, 36 con và 100 chân chẵn. Hỏi có bao nhiêu gà và chó?', "files": []},],
            [{"text": 'Hôm nay tôi có 5 quả cam. Hôm qua tôi ăn 2 quả. Vậy hôm nay tôi có mấy quả cam?', "files": []},],
            [{"text": '5 điều bác Hồ dạy là gì?', "files": []},],
            [{"text": "Tolong bantu saya menulis email ke lembaga pemerintah untuk mencari dukungan finansial untuk penelitian AI.", "files": []},],
            [{"text": "ຂໍແຈ້ງ 5 ສະຖານທີ່ທ່ອງທ່ຽວໃນນະຄອນຫຼວງວຽງຈັນ", "files": []},],
            [{"text": 'ငွေကြေးအခက်အခဲကြောင့် ပညာသင်ဆုတောင်းဖို့ တက္ကသိုလ်ကို စာတစ်စောင်ရေးပြီး ကူညီပေးပါ။', "files": []},],
            [{"text": "Sally has 3 brothers, each brother has 2 sisters. How many sister sally has?", "files": []},],
            [{"text": "There are 3 killers in a room. Someone enters the room and kills 1 of them. Assuming no one leaves the room. How many killers are left in the room?", "files": []},],
            [{"text": "Assume the laws of physics on Earth. A small marble is put into a normal cup and the cup is placed upside down on a table. Someone then takes the cup and puts it inside the microwave. Where is the ball now? Explain your reasoning step by step.", "files": []},],
            [{"text": "Why my parents did not invited me to their weddings?", "files": []},],
        ]
    
    @property
    def mm_textbox_placeholder(self):
        return "Type message or upload an image"
    
    @property
    def mm_accept_file_types(self):
        return ["image"]
    
    @property
    def gradio_fn(self):
        return vision_chat_response_stream_multiturn_engine

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

        assert MultimodalTextbox is not None

        additional_inputs = [
            gr.Number(value=temperature, label='Temperature', min_width=20), 
            gr.Number(value=max_tokens, label='Max-tokens', min_width=20), 
            gr.Textbox(value=system_prompt, label='System prompt', lines=1),
            gr.Textbox(value=IMAGE_TOKEN, label='Visual token', lines=1, interactive=IMAGE_TOKEN_INTERACTIVE, min_width=20),
        ]


        demo_chat = MultiModalTextChatInterface(
            self.gradio_fn,
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
            textbox=MultimodalTextbox(
                placeholder=self.mm_textbox_placeholder, 
                interactive=True,
                scale=9,
                show_label=False,
                # file_types=["image", '.pdf', '.docx', '.txt'],
                file_types=self.mm_accept_file_types,
            ),
            title=title,
            description=description,
            additional_inputs=additional_inputs, 
            additional_inputs_accordion=gr.Accordion("Additional Inputs", open=True),
            examples=self.examples,
            cache_examples=False,
            css=CSS,
            fill_height=True,
        )

        return demo_chat



@register_demo
class DocMMChatInterfaceDemo(VisionMMChatInterfaceDemo):
    """
    Accept vision image
    """

    @property
    def tab_name(self):
        return "Doc Chat"
    
    @property
    def mm_textbox_placeholder(self):
        return "Type message or upload a doc file (pdf, docx, txt)"
    
    @property
    def mm_accept_file_types(self):
        return ['.pdf', '.docx', '.txt']

    @property
    def examples(self):
        from pathlib import Path
        from gradio.data_classes import FileData, GradioModel
        return [
            [{"text": "Hãy giải thích thuyết tương đối rộng.", "files": []},],
            [{"text": "Hãy giải thích vấn đề P vs NP.", "files": []},],
            [{"text": "Explain general relativity.", "files": []},],
            [{"text": 'Vừa gà vừa chó, bó lại cho tròn, 36 con và 100 chân chẵn. Hỏi có bao nhiêu gà và chó?', "files": []},],
            [{"text": 'Hôm nay tôi có 5 quả cam. Hôm qua tôi ăn 2 quả. Vậy hôm nay tôi có mấy quả cam?', "files": []},],
            [{"text": '5 điều bác Hồ dạy là gì?', "files": []},],
            [{"text": "Tolong bantu saya menulis email ke lembaga pemerintah untuk mencari dukungan finansial untuk penelitian AI.", "files": []},],
            [{"text": "ຂໍແຈ້ງ 5 ສະຖານທີ່ທ່ອງທ່ຽວໃນນະຄອນຫຼວງວຽງຈັນ", "files": []},],
            [{"text": 'ငွေကြေးအခက်အခဲကြောင့် ပညာသင်ဆုတောင်းဖို့ တက္ကသိုလ်ကို စာတစ်စောင်ရေးပြီး ကူညီပေးပါ။', "files": []},],
            [{"text": "Sally has 3 brothers, each brother has 2 sisters. How many sister sally has?", "files": []},],
            [{"text": "There are 3 killers in a room. Someone enters the room and kills 1 of them. Assuming no one leaves the room. How many killers are left in the room?", "files": []},],
            [{"text": "Assume the laws of physics on Earth. A small marble is put into a normal cup and the cup is placed upside down on a table. Someone then takes the cup and puts it inside the microwave. Where is the ball now? Explain your reasoning step by step.", "files": []},],
            [{"text": "Why my parents did not invited me to their weddings?", "files": []},],
        ]
    
    @property
    def gradio_fn(self):
        # return vision_chat_response_stream_multiturn_engine
        return doc_chat_response_stream_multiturn_engine
    




@register_demo
class VisionDocMMChatInterfaceDemo(VisionMMChatInterfaceDemo):
    """
    Accept vision image
    """

    @property
    def tab_name(self):
        return "Vision Doc Chat"
    
    @property
    def mm_textbox_placeholder(self):
        return "Type message or upload an image or doc file (pdf, docx, txt)"
    
    @property
    def mm_accept_file_types(self):
        return ['image', '.pdf', '.docx', '.txt']
    
    @property
    def gradio_fn(self):
        # return vision_chat_response_stream_multiturn_engine
        return vision_doc_chat_response_stream_multiturn_engine
    




