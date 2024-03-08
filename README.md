# Multi-purpose Chatbot (Local, Remote and HF spaces)

A Chatbot UI that support Chatbot, RAG, Text completion, Multi-modal across [HF Transformers](https://github.com/huggingface/transformers), [llama.cppp](https://github.com/ggerganov/llama.cpp), [Apple MLX](https://github.com/ml-explore/mlx) and [vLLM](https://github.com/vllm-project/vllm).

Designed support both locally, remote and huggingface spaces.

![image](assets/image_doc_rag.gif)

---

**Checkout cool demos using Multi-purpose chatbot.**
- [MultiModal SeaLLMs/SeaLLM-7B](https://huggingface.co/spaces/SeaLLMs/SeaLLM-7B)
- [MultiPurpose-Chatbot-DEMO test](https://huggingface.co/spaces/nxphi47/MultiPurpose-Chatbot-DEMO) - This DEMO test the UI without LLM.

**Supported features**
- Vanilla chat interface - [ChatInterfaceDemo](multipurpose_chatbot/demos/chat_interface.py)
- Chat with short document (full context) - [DocChatInterfaceDemo](multipurpose_chatbot/demos/multimodal_chat_interface.py)
- Chat with visual image - [VisionChatInterfaceDemo](multipurpose_chatbot/demos/multimodal_chat_interface.py)
- Chat with visual image and short document - [VisionDocChatInterfaceDemo](multipurpose_chatbot/demos/multimodal_chat_interface.py)
- Chat with long document via RAG - [RagChatInterfaceDemo](multipurpose_chatbot/demos/rag_chat_interface.py)
- Text completion (free form prompting) - [TextCompletionDemo](multipurpose_chatbot/demos/text_completion.py)
- Batch inference (via file upload with vLLM) - [BatchInferenceDemo](multipurpose_chatbot/demos/batch_inference.py)

**Support backend**
- [GPU Transformers](https://github.com/huggingface/transformers) with full support MultiModal, document QA, RAG, completion.
- [llama.cppp](https://github.com/ggerganov/llama.cpp) like Transformers, except pending MultiModal. PR welcome.
- [Apple MLX](https://github.com/ml-explore/mlx) like Transformers, except pending MultiModal. PR welcome.
- [vLLM](https://github.com/vllm-project/vllm) like Transformers + **Batch inference via file upload**, pending MultiModal. PR welcome.

Multi-purpose Chatbot use `ENVIRONMENT VARIABLE` instead of `argparse` to set hyperparmeters to support seamless integration with HF space, which requires us to set params via environment vars. The app is launch only with `python app.py`

## Installation

```bash
pip install -r requirements.txt
```

#### Transformers
```bash
pip install -r transformers_requirements.txt
```


#### VLLM
```bash
pip install -r vllm_requirements.txt
```


#### llama.cpp
Follow [Llama-cpp-python](https://llama-cpp-python.readthedocs.io/en/latest/#installation) to install `llama.cpp`

e.g: On Macos
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```


#### MLX

Only on MacOS, remember to install [**NATIVE** python environment](https://ml-explore.github.io/mlx/build/html/install.html).

```bash
python -c "import platform; print(platform.processor())"
# should output "arm", if not reinstall python with native
```

Install requirements
```bash
pip install -r mlx_requirements.txt
```


## Usage

We use bash environment to define model variables

#### Transformers

`MODEL_PATH` must be a model with chat_template with system prompt (e.g Mistral-7B-Instruct-v0.2 does not have system prompt)

```bash
export BACKEND=transformers
export MODEL_PATH=teknium/OpenHermes-2.5-Mistral-7B
export RAG_EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
export DEMOS=DocChatInterfaceDemo,ChatInterfaceDemo,RagChatInterfaceDemo,TextCompletionDemo
python app.py
```

#### Llava-1.5 Transformers


```bash
export CUDA_VISIBLE_DEVICES=0
export TEMPERATURE=0.7
export MAX_TOKENS=512
export MODEL_PATH=llava-hf/llava-1.5-7b-hf
export IMAGE_TOKEN="<image>"
export BACKEND=llava15_transformers
export DEMOS=VisionChatInterfaceDemo,VisionDocChatInterfaceDemo,TextCompletionDemo
python app.py

```

#### VLLM

```bash
export BACKEND=vllm
export MODEL_PATH=teknium/OpenHermes-2.5-Mistral-7B
export RAG_EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
export DEMOS=DocChatInterfaceDemo,ChatInterfaceDemo,RagChatInterfaceDemo,TextCompletionDemo
python app.py
```


#### llama.cpp

```bash
export BACKEND=llama_cpp
export MODEL_PATH=/path/to/model.gguf
export RAG_EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
export DEMOS=DocChatInterfaceDemo,ChatInterfaceDemo,RagChatInterfaceDemo,TextCompletionDemo
python app.py
```


#### MLX

```bash
export BACKEND=mlx
export MODEL_PATH=mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX
export RAG_EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
export DEMOS=DocChatInterfaceDemo,ChatInterfaceDemo,RagChatInterfaceDemo,TextCompletionDemo
python app.py
```


## Customization

#### Configs:

* [configs.py](multipurpose_chatbot/configs.py) where you can find customize UI markdowns and settings global variables

#### Backend and engines
* [base_engine](multipurpose_chatbot/engines/base_engine.py),  [transformers_engine](multipurpose_chatbot/engines/transformers_engine.py) and [llama_cpp_engine](multipurpose_chatbot/engines/llama_cpp_engine.py) to find how different model backend works. Feel free to extend and implement new features.
* [llava15_transformers_engine](multipurpose_chatbot/engines/llava15_transformers_engine.py) describe how to implement Llava-1.5


#### Gradio Demo tabs
* Checkout [chat_interface](multipurpose_chatbot/demos/chat_interface.py), [multimodal_chat_interface](multipurpose_chatbot/demos/multimodal_chat_interface.py) and other interface demo under [multipurpose_chatbot/demos](multipurpose_chatbot/demos) to find out how the demo works.


#### Enableing demos

Setting comma-separated demo class names (e.g `ChatInterfaceDemo` to enable demo).

```bash
export DEMOS=VisionDocChatInterfaceDemo,VisionChatInterfaceDemo,DocChatInterfaceDemo,ChatInterfaceDemo,RagChatInterfaceDemo,TextCompletionDemo
```


## Contributing

We welcome and value any contributions and collaborations. Feel free to open a PR


## Citation

If you find our project useful, hope you can star our repo and cite our repo as follows:
```
@article{multipurpose_chatbot_2024,
  author = {Xuan-Phi Nguyen, },
  title = {Multipurpose Chatbot},
  year = 2024,
}
