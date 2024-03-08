# Multi-purpose Chatbot (Local, Remote and HF spaces)

A Chatbot UI that support Chatbot, RAG, Text completion, Multi-modal across [HF Transformers](https://github.com/huggingface/transformers), [llama.cppp](https://github.com/ggerganov/llama.cpp), [Apple MLX](https://github.com/ml-explore/mlx) and [vLLM](https://github.com/vllm-project/vllm).

Designed support both locally, remote and huggingface spaces.

![image](assets/image_doc_rag.gif)

---

**Checkout cool demos using Multi-purpose chatbot.**
- [MultiModal SeaLLM-7B](https://huggingface.co/spaces/SeaLLMs/SeaLLM-7B)

**Supported features**



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
python app.py
```


#### VLLM

```bash
export BACKEND=vllm
export MODEL_PATH=teknium/OpenHermes-2.5-Mistral-7B
python app.py
```


#### llama.cpp

```bash
export BACKEND=llama_cpp
export MODEL_PATH=/path/to/model.gguf
python app.py
```


#### MLX

```bash
export BACKEND=mlx
export MODEL_PATH=mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX
python app.py
```


## Customization



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