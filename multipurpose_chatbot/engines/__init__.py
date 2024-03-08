
from .base_engine import BaseEngine

BACKENDS = [
    "mlx", 
    "vllm", 
    "transformers",
    "llava15_transformers",
    "llama_cpp",
    # "llava_llama_cpp",
    "debug",
]

ENGINE_LOADED = False


def load_multipurpose_chatbot_engine(backend: str):
    # ! lazy import other engines
    global ENGINE_LOADED
    assert backend in BACKENDS, f'{backend} not in {BACKENDS}'
    if ENGINE_LOADED:
        raise RuntimeError(f'{ENGINE_LOADED=} this means load_multipurpose_chatbot_engine has already been called! Check your codes.')
    print(f'Load model from {backend}')
    if backend == "mlx":
        from .mlx_engine import MlxEngine
        model_engine = MlxEngine()
    elif backend == 'vllm':
        from .vllm_engine import VllmEngine
        model_engine = VllmEngine()
    elif backend == 'transformers':
        from .transformers_engine import TransformersEngine
        model_engine = TransformersEngine()
    elif backend == 'llava15_transformers':
        from .llava15_transformers_engine import Llava15TransformersEngine
        model_engine = Llava15TransformersEngine()
    elif backend == 'llama_cpp':
        from .llama_cpp_engine import LlamaCppEngine
        model_engine = LlamaCppEngine()
    # ! llava_llama_cpp currently not done due to bugs
    # elif backend == 'llava_llama_cpp':
    #     from .llava_llama_cpp_engine import LlavaLlamaCppEngine
    #     model_engine = LlavaLlamaCppEngine()
    elif backend == 'debug':
        from .debug_engine import DebugEngine
        model_engine = DebugEngine()
    else:
        raise ValueError(f'backend invalid: {BACKENDS} vs {backend}')

    model_engine.load_model()
    ENGINE_LOADED = True
    return model_engine
    # ! add more llama.cpp engine here.


