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
from typing import AsyncGenerator, Callable, Literal, Union, cast

from gradio_client.documentation import document, set_documentation_group
from gradio.components import Button, Component
from gradio.events import Dependency, EventListenerMethod
from typing import List, Optional, Union, Dict, Tuple
from tqdm.auto import tqdm
from huggingface_hub import snapshot_download


def create_class_func_registry():
    registry = {}
    def register_registry(cls, exist_ok=False):
        assert exist_ok or cls.__name__ not in registry, f'{cls} already in registry: {registry}'
        registry[cls.__name__] = cls
        return cls
        
    def get_registry(name):
        assert name in registry, f'{name} not in registry: {registry}'
        return registry[name]

    return registry, register_registry, get_registry

DEMOS, register_demo, get_demo_class = create_class_func_registry()


class BaseDemo(object):
    """
    All demo should be created from BaseDemo and registered with @register_demo
    """
    def __init__(self) -> None:
        pass

    def create_demo(
            self, 
            title: Optional[str] = None, 
            description: Optional[str] = None,
            **kwargs,
    ) -> gr.Blocks:
        pass


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

