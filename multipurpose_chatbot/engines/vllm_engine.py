import os
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

from typing import List, Optional, Union, Dict, Tuple
from tqdm.auto import tqdm
from huggingface_hub import snapshot_download

from gradio.components import Button
from gradio.events import Dependency, EventListenerMethod

from .base_engine import BaseEngine
# @@ environments ================

from ..configs import (
    DTYPE,
    TENSOR_PARALLEL,
    MODEL_PATH,
    QUANTIZATION,
    MAX_TOKENS,
    TEMPERATURE,
    FREQUENCE_PENALTY,
    PRESENCE_PENALTY,
    GPU_MEMORY_UTILIZATION,
    STREAM_CHECK_MULTIPLE,
    STREAM_YIELD_MULTIPLE,

)


llm = None
demo = None



def vllm_abort(self):
    sh = self.llm_engine.scheduler
    for g in (sh.waiting + sh.running + sh.swapped):
        sh.abort_seq_group(g.request_id)
    from vllm.sequence import SequenceStatus
    scheduler = self.llm_engine.scheduler
    for state_queue in [scheduler.waiting, scheduler.running, scheduler.swapped]:
        for seq_group in state_queue:
            # if seq_group.request_id == request_id:
            # Remove the sequence group from the state queue.
            state_queue.remove(seq_group)
            for seq in seq_group.seqs:
                if seq.is_finished():
                    continue
                scheduler.free_seq(seq, SequenceStatus.FINISHED_ABORTED)


def _vllm_run_engine(self: Any, use_tqdm: bool = False) -> Dict[str, Any]:
    from vllm.outputs import RequestOutput
    # Initialize tqdm.
    if use_tqdm:
        num_requests = self.llm_engine.get_num_unfinished_requests()
        pbar = tqdm(total=num_requests, desc="Processed prompts")
    # Run the engine.
    outputs: Dict[str, RequestOutput] = {}
    while self.llm_engine.has_unfinished_requests():
        step_outputs = self.llm_engine.step()
        for output in step_outputs:
            outputs[output.request_id] = output
        if len(outputs) > 0:
            yield outputs


def vllm_generate_stream(
    self: Any,
    prompts: Optional[Union[str, List[str]]] = None,
    sampling_params: Optional[Any] = None,
    prompt_token_ids: Optional[List[List[int]]] = None,
    use_tqdm: bool = False,
) -> Dict[str, Any]:
    """Generates the completions for the input prompts.

    NOTE: This class automatically batches the given prompts, considering
    the memory constraint. For the best performance, put all of your prompts
    into a single list and pass it to this method.

    Args:
        prompts: A list of prompts to generate completions for.
        sampling_params: The sampling parameters for text generation. If
            None, we use the default sampling parameters.
        prompt_token_ids: A list of token IDs for the prompts. If None, we
            use the tokenizer to convert the prompts to token IDs.
        use_tqdm: Whether to use tqdm to display the progress bar.

    Returns:
        A list of `RequestOutput` objects containing the generated
        completions in the same order as the input prompts.
    """
    from vllm import LLM, SamplingParams
    if prompts is None and prompt_token_ids is None:
        raise ValueError("Either prompts or prompt_token_ids must be "
                            "provided.")
    if isinstance(prompts, str):
        # Convert a single prompt to a list.
        prompts = [prompts]
    if prompts is not None and prompt_token_ids is not None:
        if len(prompts) != len(prompt_token_ids):
            raise ValueError("The lengths of prompts and prompt_token_ids "
                                "must be the same.")
    if sampling_params is None:
        # Use default sampling params.
        sampling_params = SamplingParams()
    # Add requests to the engine.
    if prompts is not None:
        num_requests = len(prompts)
    else:
        num_requests = len(prompt_token_ids)
    for i in range(num_requests):
        prompt = prompts[i] if prompts is not None else None
        if prompt_token_ids is None:
            token_ids = None
        else:
            token_ids = prompt_token_ids[i]
        self._add_request(prompt, sampling_params, token_ids)
    # return self._run_engine(use_tqdm)
    yield from _vllm_run_engine(self, use_tqdm)



class VllmEngine(BaseEngine):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @property
    def tokenizer(self):
        return self._model.get_tokenizer()

    def load_model(self, ):
        import torch
        try:
            compute_capability = torch.cuda.get_device_capability()
            print(f'Torch CUDA compute_capability: {compute_capability}')
        except Exception as e:
            print(f'Failed to print compute_capability version: {e}')

        import vllm
        from vllm import LLM

        print(f'VLLM: {vllm.__version__=}')

        if QUANTIZATION == 'awq':
            print(F'Load model in int4 quantization')
            llm = LLM(
                model=MODEL_PATH, 
                dtype="float16", 
                tensor_parallel_size=TENSOR_PARALLEL, 
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION, 
                quantization="awq", 
                max_model_len=MAX_TOKENS
            )
        else:
            llm = LLM(
                model=MODEL_PATH, 
                dtype=DTYPE, 
                tensor_parallel_size=TENSOR_PARALLEL, 
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION, 
                max_model_len=MAX_TOKENS
            )

        try:
            print(llm.llm_engine.workers[0].model)
        except Exception as e:
            print(f'Cannot print model worker: {e}')

        try:
            llm.llm_engine.scheduler_config.max_model_len = MAX_TOKENS
            llm.llm_engine.scheduler_config.max_num_batched_tokens = MAX_TOKENS
        except Exception as e:
            print(f'Cannot set parameters: {e}')

        self._model = llm

    def generate_yield_string(self, prompt, temperature, max_tokens, stop_strings: Optional[Tuple[str]] = None, **kwargs):
        from vllm import SamplingParams
        # ! must abort previous ones
        vllm_abort(self._model)
        sampling_params = SamplingParams(
            temperature=temperature, 
            max_tokens=max_tokens,
            # frequency_penalty=frequency_penalty,
            # presence_penalty=presence_penalty,
            stop=stop_strings,
        )
        cur_out = None
        num_tokens = len(self.tokenizer.encode(prompt))
        for j, gen in enumerate(vllm_generate_stream(self._model, prompt, sampling_params)):
            if cur_out is not None and (STREAM_YIELD_MULTIPLE < 1 or j % STREAM_YIELD_MULTIPLE == 0) and j > 0:
                yield cur_out, num_tokens
            assert len(gen) == 1, f'{gen}'
            item = next(iter(gen.values()))
            cur_out = item.outputs[0].text

        if cur_out is not None:
            full_text = prompt + cur_out
            num_tokens = len(self.tokenizer.encode(full_text))
            yield cur_out, num_tokens
    
    def batch_generate(self, prompts, temperature, max_tokens, stop_strings: Optional[Tuple[str]] = None, **kwargs):
        """
        Only vllm should support this, the other engines is only batch=1 only
        """
        from vllm import SamplingParams
        # ! must abort previous ones
        vllm_abort(self._model)
        sampling_params = SamplingParams(
            temperature=temperature, 
            max_tokens=max_tokens,
            # frequency_penalty=frequency_penalty,
            # presence_penalty=presence_penalty,
            stop=stop_strings,
        )
        generated = self._model.generate(prompts, sampling_params, use_tqdm=False)
        responses = [g.outputs[0].text for g in generated]
        return responses