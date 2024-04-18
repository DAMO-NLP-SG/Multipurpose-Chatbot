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


from .base_demo import register_demo, get_demo_class, BaseDemo
from ..configs import (
    SYSTEM_PROMPT,
    MODEL_NAME,
    MAX_TOKENS,
    TEMPERATURE,
    USE_PANEL,
    CHATBOT_HEIGHT,
)

from ..globals import MODEL_ENGINE

# CHAT_EXAMPLES = [
# ]

CHAT_EXAMPLES = [
    ["Hãy giải thích thuyết tương đối rộng."],
    ["Hãy giải thích vấn đề P vs NP."],
    ["Explain general relativity."],
    ['Vừa gà vừa chó, bó lại cho tròn, 36 con và 100 chân chẵn. Hỏi có bao nhiêu gà và chó?'],
    ['Hôm nay tôi có 5 quả cam. Hôm qua tôi ăn 2 quả. Vậy hôm nay tôi có mấy quả cam?'],
    ['5 điều bác Hồ dạy là gì?'],
    ["Tolong bantu saya menulis email ke lembaga pemerintah untuk mencari dukungan finansial untuk penelitian AI."],
    ["ຂໍແຈ້ງ 5 ສະຖານທີ່ທ່ອງທ່ຽວໃນນະຄອນຫຼວງວຽງຈັນ"],
    ['ငွေကြေးအခက်အခဲကြောင့် ပညာသင်ဆုတောင်းဖို့ တက္ကသိုလ်ကို စာတစ်စောင်ရေးပြီး ကူညီပေးပါ။'],
    ["Sally has 3 brothers, each brother has 2 sisters. How many sister sally has? Explain step by step."],
    ["There are 3 killers in a room. Someone enters the room and kills 1 of them. Assuming no one leaves the room. How many killers are left in the room?"],
    ["Assume the laws of physics on Earth. A small marble is put into a normal cup and the cup is placed upside down on a table. Someone then takes the cup and puts it inside the microwave. Where is the ball now? Explain your reasoning step by step."],
    ["Why didn't my parents invite me to their weddings?"],
]

DATETIME_FORMAT = "Current date time: {cur_datetime}."


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



def get_datetime_string():
    from datetime import datetime
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%B %d, %Y, %H:%M:%S")
    return dt_string


def format_conversation(history, system_prompt=None):
    _str = '\n'.join([
        (
            f'<<<User>>> {h[0]}\n'
            f'<<<Asst>>> {h[1]}'
        )
        for h in history
    ])
    _str = ""
    for mes, res in history:
        if mes is not None:
            _str += f'<<<User>>> {mes}\n'
        if res is not None:
            _str += f'<<<Asst>>> {res}\n'
    if system_prompt is not None:
        _str = f"<<<Syst>>> {system_prompt}\n" + _str
    return _str

    
def chat_response_stream_multiturn_engine(
    message: str, 
    history: List[Tuple[str, str]], 
    temperature: float, 
    max_tokens: int, 
    system_prompt: Optional[str] = SYSTEM_PROMPT,
):
    global MODEL_ENGINE
    temperature = float(temperature)
    # ! remove frequency_penalty
    # frequency_penalty = float(frequency_penalty)
    max_tokens = int(max_tokens)
    message = message.strip()
    if len(message) == 0:
        raise gr.Error("The message cannot be empty!")
    # ! skip safety
    if DATETIME_FORMAT in system_prompt:
        # ! This sometime works sometimes dont
        system_prompt = system_prompt.format(cur_datetime=get_datetime_string())
    full_prompt = gradio_history_to_conversation_prompt(message.strip(), history=history, system_prompt=system_prompt)
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
    )):
        if isinstance(outputs, tuple):
            response, num_tokens = outputs
        else:
            response, num_tokens = outputs, -1
        yield response, num_tokens
        
    print(format_conversation(history + [[message, response]]))

    if response is not None:
        yield response, num_tokens


class CustomizedChatInterface(gr.ChatInterface):
    """
    Fixing some issue with chatinterace
    """
    
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

            if examples:
                if self.is_generator:
                    examples_fn = self._examples_stream_fn
                else:
                    examples_fn = self._examples_fn

                self.examples_handler = Examples(
                    examples=examples,
                    inputs=[self.textbox] + self.additional_inputs,
                    # outputs=self.chatbot,
                    # fn=examples_fn,
                    cache_examples=False,
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
                .then(
                    submit_fn,
                    [self.saved_input, self.chatbot_state] + self.additional_inputs,
                    [self.chatbot, self.chatbot_state, self.num_tokens],
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
    
    def _delete_prev_fn(
        self,
        message: str | dict[str, list],
        history: list[list[str | tuple | None]],
    ) -> tuple[
        list[list[str | tuple | None]],
        str | dict[str, list],
        list[list[str | tuple | None]],
    ]:
        if isinstance(message, dict):
            # handling for multi-modal
            remove_input = (
                len(message["files"]) + 1
                if message["text"] is not None
                else len(message["files"])
            )
            history = history[:-remove_input]
        else:
            history = history[:-1]
        return history, message or "", history

    def _clear_and_save_textbox(self, message: str) -> tuple[str, str]:
        return "", message

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

@register_demo
class ChatInterfaceDemo(BaseDemo):
    @property
    def tab_name(self):
        return "Chat"
    
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

        demo_chat = CustomizedChatInterface(
            chat_response_stream_multiturn_engine,
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
            title=title,
            description=description,
            additional_inputs=[
                gr.Number(value=temperature, label='Temperature (higher -> more random)'), 
                gr.Number(value=max_tokens, label='Max generated tokens (increase if want more generation)'), 
                # gr.Number(value=frequence_penalty, label='Frequency penalty (> 0 encourage new tokens over repeated tokens)'), 
                # gr.Number(value=presence_penalty, label='Presence penalty (> 0 encourage new tokens, < 0 encourage existing tokens)'), 
                gr.Textbox(value=system_prompt, label='System prompt', lines=4)
            ], 
            examples=CHAT_EXAMPLES,
            cache_examples=False
        )
        return demo_chat

