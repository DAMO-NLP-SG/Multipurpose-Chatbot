
import os

# ! UI Markdown information

MODEL_TITLE = "<h1>Multi-Purpose Chatbot</h1>"

MODEL_DESC = f"""
<div style='display:flex; gap: 0.25rem; '>
<a href='https://github.com/DAMO-NLP-SG/Multipurpose-Chatbot'><img src='https://img.shields.io/badge/Github-Code-success'></a>
</div>
<span style="font-size: larger">
A multi-purpose helpful assistant with multiple functionalities (Chat, text-completion, RAG chat, batch inference).
</span>
""".strip()



MODEL_INFO = """
<h4>Model Name: {model_path}</h4>
"""

CITE_MARKDOWN = """
## Citation
If you find our project useful, hope you can star our repo and cite our repo as follows:
```
@article{multipurpose_chatbot_2024,
  author = {Xuan-Phi Nguyen, },
  title = {Multipurpose Chatbot},
  year = 2024,
}
```
"""

USE_PANEL = bool(int(os.environ.get("USE_PANEL", "1")))
CHATBOT_HEIGHT = int(os.environ.get("CHATBOT_HEIGHT", "500"))

ALLOWED_PATHS = []


DEMOS = os.environ.get("DEMOS", "")

DEMOS = DEMOS.split(",") if DEMOS.strip() != "" else [
    "DocChatInterfaceDemo",
    "ChatInterfaceDemo",
    "TextCompletionDemo",
    # "RagChatInterfaceDemo",
    # "VisionChatInterfaceDemo",
    # "VisionDocChatInterfaceDemo",
]

# DEMOS=VisionDocChatInterfaceDemo,DocChatInterfaceDemo,ChatInterfaceDemo,RagChatInterfaceDemo,TextCompletionDemo



# ! server info

PORT = int(os.environ.get("PORT", "7860"))
PROXY = os.environ.get("PROXY", "").strip()

# ! backend info

BACKEND = os.environ.get("BACKEND", "debug")

# ! model information
# for RAG
RAG_EMBED_MODEL_NAME = os.environ.get("RAG_EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_SIZE", "50"))


SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", """You are a helpful, respectful, honest and safe AI assistant.""")

MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2048"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
# ! these values currently not used
FREQUENCE_PENALTY = float(os.environ.get("FREQUENCE_PENALTY", "0.0"))
PRESENCE_PENALTY = float(os.environ.get("PRESENCE_PENALTY", "0.0"))


# Transformers or vllm
MODEL_PATH = os.environ.get("MODEL_PATH", "teknium/OpenHermes-2.5-Mistral-7B")
MODEL_NAME = os.environ.get("MODEL_NAME", "Cool-Chatbot")
DTYPE = os.environ.get("DTYPE", "bfloat16")
DEVICE = os.environ.get("DEVICE", "cuda")

# VLLM
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.9"))
TENSOR_PARALLEL = int(os.environ.get("TENSOR_PARALLEL", "1"))
QUANTIZATION = str(os.environ.get("QUANTIZATION", ""))
STREAM_YIELD_MULTIPLE = int(os.environ.get("STREAM_YIELD_MULTIPLE", "1"))
# how many iterations to perform safety check on response
STREAM_CHECK_MULTIPLE = int(os.environ.get("STREAM_CHECK_MULTIPLE", "0"))

# llama.cpp
DEFAULT_CHAT_TEMPLATE = os.environ.get("DEFAULT_CHAT_TEMPLATE", "chatml")
N_CTX = int(os.environ.get("N_CTX", "4096"))
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "-1"))

# llava.llama.cpp
# ! pending development

# Multimodal
# IMAGE_TOKEN = os.environ.get("IMAGE_TOKEN", "[IMAGE]<|image|>[/IMAGE]")
IMAGE_TOKEN = os.environ.get("IMAGE_TOKEN", "<image>")
IMAGE_TOKEN_INTERACTIVE = bool(int(os.environ.get("IMAGE_TOKEN_INTERACTIVE", "0")))
# ! IMAGE_TOKEN_LENGTH expected embedding lengths of an image to calculate the actual tokens
IMAGE_TOKEN_LENGTH = int(os.environ.get("IMAGE_TOKEN_LENGTH", "576"))
# ! Llava1.6 to calculate the maximum number of patches in an image (max=5 for Llava1.6)
MAX_PACHES = int(os.environ.get("MAX_PACHES", "1"))
