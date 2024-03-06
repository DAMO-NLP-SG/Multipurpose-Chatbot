# Multi-purpose Chatbot Serving (Remote and Locally)

Support Chatbot, RAG, Text completion, Multi-modal


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

Only on Macos, remember to install [**NATIVE** python environment](https://ml-explore.github.io/mlx/build/html/install.html).

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

```bash
export BACKEND=transformers
export MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.2
python app.py
```


#### VLLM

```bash
export BACKEND=vllm
export MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.2
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
export MODEL_PATH=mlx-community/Mistral-7B-Instruct-v0.2-4bit-mlx
python app.py
```


## Customization



## Conclusion