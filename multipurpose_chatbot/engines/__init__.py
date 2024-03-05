
from .base_engine import BaseEngine

BACKENDS = [
    "mlx", 
    "vllm", 
    "transformers",
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
    elif backend == 'debug':
        from .debug_engine import DebugEngine
        model_engine = DebugEngine()

    model_engine.load_model()
    ENGINE_LOADED = True
    return model_engine
    # ! add more llama.cpp engine here.


