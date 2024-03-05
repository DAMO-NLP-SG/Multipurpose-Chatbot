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

from .base_engine import BaseEngine

# ! Remember to use static cache

class TransformersEngine(BaseEngine):
    pass