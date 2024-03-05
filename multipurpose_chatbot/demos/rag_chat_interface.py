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


from .base_demo import register_demo, get_demo_class, BaseDemo


# SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", """You are a helpful, respectful, honest and safe AI assistant.""")
# MODEL_NAME = os.environ.get("MODEL_NAME", "Cool-Chatbot")
# MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2048"))
# MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2048"))
# TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.1"))

from ..globals import MODEL_ENGINE, RAG_CURRENT_FILE, RAG_EMBED, load_embeddings, get_rag_embeddings

from .chat_interface import (
    SYSTEM_PROMPT,
    MODEL_NAME,
    MAX_TOKENS,
    TEMPERATURE,
    CHAT_EXAMPLES,
    gradio_history_to_openai_conversations,
    gradio_history_to_conversation_prompt,
    DATETIME_FORMAT,
    get_datetime_string,
    format_conversation,
    chat_response_stream_multiturn_engine,
    ChatInterfaceDemo,
    CustomizedChatInterface,
)

RAG_CURRENT_VECTORSTORE = None

def load_document_split_vectorstore(file_path):
    global RAG_CURRENT_FILE, RAG_EMBED, RAG_CURRENT_VECTORSTORE
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
    from langchain_community.vectorstores import Chroma, FAISS
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
    # assert RAG_EMBED is not None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    splits = loader.load_and_split(splitter)
    # RAG_CURRENT_VECTORSTORE = Chroma.from_documents(documents=splits, embedding=RAG_EMBED)
    # RAG_CURRENT_VECTORSTORE = None
    # RAG_CURRENT_VECTORSTORE = Chroma.from_documents(documents=splits, embedding=get_rag_embeddings())
    RAG_CURRENT_VECTORSTORE = FAISS.from_texts(texts=[s.page_content for s in splits], embedding=get_rag_embeddings())
    return RAG_CURRENT_VECTORSTORE



def docs_to_rag_context(docs: List[str]):
    contexts = "\n".join([d.page_content for d in docs])
    context = f"""### Begin document
{contexts}
### End document
Asnwer the following query exclusively based on the information provided in the document above. \
Remember to follow the language of the user query.
"""
    return context


def maybe_get_doc_context(message, file_input, rag_num_docs: Optional[int] = 3):
    doc_context = None
    if file_input is not None:
        if file_input == RAG_CURRENT_FILE:
            # reuse
            vectorstore = RAG_CURRENT_VECTORSTORE
            print(f'Reuse vectorstore: {file_input}')
        else:
            vectorstore = load_document_split_vectorstore(file_input)
            print(f'New vectorstore: {RAG_CURRENT_FILE} {file_input}')
            RAG_CURRENT_FILE = file_input
        docs = vectorstore.similarity_search(message, k=rag_num_docs)
        doc_context = docs_to_rag_context(docs)
    return doc_context


def chat_response_stream_multiturn_doc_engine(
    message: str, 
    history: List[Tuple[str, str]], 
    file_input: Optional[str] = None,
    temperature: float = 0.7, 
    max_tokens: int = 1024, 
    # frequency_penalty: float,
    # presence_penalty: float,
    system_prompt: Optional[str] = SYSTEM_PROMPT,
    # profile: Optional[gr.OAuthProfile] = None,
    rag_num_docs: Optional[int] = 3,
):
    global MODEL_ENGINE, RAG_CURRENT_FILE, RAG_EMBED, RAG_CURRENT_VECTORSTORE
    if len(message) == 0:
        raise gr.Error("The message cannot be empty!")
    
    rag_num_docs = int(rag_num_docs)
    doc_context = None
    if file_input is not None:
        if file_input == RAG_CURRENT_FILE:
            # reuse
            vectorstore = RAG_CURRENT_VECTORSTORE
            print(f'Reuse vectorstore: {file_input}')
        else:
            vectorstore = load_document_split_vectorstore(file_input)
            print(f'New vectorstore: {RAG_CURRENT_FILE} {file_input}')
            RAG_CURRENT_FILE = file_input
        docs = vectorstore.similarity_search(message, k=rag_num_docs)
        doc_context = docs_to_rag_context(docs)
    
    if doc_context is not None:
        message = f"{doc_context}\n\n{message}"
    
    yield from chat_response_stream_multiturn_engine(
        message, history, temperature, max_tokens, system_prompt
    )


@register_demo
class RagChatInterfaceDemo(ChatInterfaceDemo):
    def create_demo(
            self, 
            title: str | None = None, 
            description: str | None = None, 
            **kwargs
        ) -> gr.Blocks:
        load_embeddings()
        global RAG_EMBED
        # assert RAG_EMBED is not None
        print(F'{RAG_EMBED=}')
        system_prompt = kwargs.get("system_prompt", SYSTEM_PROMPT)
        max_tokens = kwargs.get("max_tokens", MAX_TOKENS)
        temperature = kwargs.get("temperature", TEMPERATURE)
        model_name = kwargs.get("model_name", MODEL_NAME)
        # frequence_penalty = FREQUENCE_PENALTY
        # presence_penalty = PRESENCE_PENALTY

        demo_chat = CustomizedChatInterface(
            chat_response_stream_multiturn_doc_engine,
            chatbot=gr.Chatbot(
                label=MODEL_NAME,
                bubble_full_width=False,
                latex_delimiters=[
                    { "left": "$", "right": "$", "display": False},
                    { "left": "$$", "right": "$$", "display": True},
                ],
                show_copy_button=True,
            ),
            # textbox=gr.Textbox(placeholder='Type message', lines=4, max_lines=128, min_width=200),
            # textbox=gr.Textbox(placeholder='Type message', lines=1, max_lines=128, min_width=200),
            submit_btn=gr.Button(value='Submit', variant="primary", scale=0),
            # ! consider preventing the stop button
            # stop_btn=None,
            title=title,
            description=description,
            additional_inputs=[
                # gr.Radio(choices=['No use RAG', 'Use RAG'], value='No use RAG', label='Use RAG for long doc, No RAG for short doc'),
                # gr.File(label='Upload Document', file_count='single', file_types=['pdf', 'docx', 'txt', 'json']),
                gr.File(label='Upload Document', file_count='single', file_types=['pdf', 'docx', 'txt']),
                gr.Number(value=temperature, label='Temperature (higher -> more random)'), 
                gr.Number(value=max_tokens, label='Max generated tokens (increase if want more generation)'), 
                # gr.Number(value=frequence_penalty, label='Frequency penalty (> 0 encourage new tokens over repeated tokens)'), 
                # gr.Number(value=presence_penalty, label='Presence penalty (> 0 encourage new tokens, < 0 encourage existing tokens)'), 
                # gr.Number(value=0, label='current_time', visible=False), 
                # ! Remove the system prompt textbox to avoid jailbreaking
                gr.Textbox(value=system_prompt, label='System prompt', lines=2),
                gr.Number(value=3, label='RAG Top-K'),
            ], 
            additional_inputs_accordion=gr.Accordion("Additional Inputs", open=True),
            # examples=CHAT_EXAMPLES,
            cache_examples=False,
        )
        return demo_chat