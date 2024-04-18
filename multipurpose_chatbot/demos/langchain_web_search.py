
import torch
import os
langchain_install_guide = """pip install --upgrade langchain langchain-community"""
try:
    from langchain_core.agents import AgentAction, AgentFinish
    from typing import List, Optional, Any, Mapping, Union, Dict, Type
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models.llms import BaseLLM
    from langchain_core.outputs import Generation, LLMResult

    from langchain_community.llms import HuggingFaceHub
    from langchain_community.llms.huggingface_hub import HuggingFaceHub
    from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline, VALID_TASKS
    from langchain_community.chat_models.huggingface import ChatHuggingFace
    from langchain_core.pydantic_v1 import root_validator

    # react style prompt
    from langchain import hub
    from langchain.agents import AgentExecutor, load_tools
    from langchain.agents.format_scratchpad import format_log_to_str
    from langchain.agents.output_parsers import (
        ReActJsonSingleInputOutputParser, ToolsAgentOutputParser 
    )
    from langchain.tools.render import render_text_description, render_text_description_and_args
    from langchain_community.utilities import SerpAPIWrapper
    from langchain_core.prompts import ChatPromptTemplate
    

    from langchain_core.utils.function_calling import (
        convert_to_openai_function,
        convert_to_openai_tool,
    )
    from langchain_core.exceptions import OutputParserException
    from langchain_core.agents import AgentAction, AgentFinish

    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
    )
    from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
    from langchain_core.pydantic_v1 import root_validator
    from langchain_core.pydantic_v1 import Extra
    # from langchain_community.tools.tavily_search import TavilySearchResults
    # from langchain_community.tools.tavily_search import (
    #     TavilySearchResults,
    #     TavilySearchAPIWrapper,
    #     Type,
    #     TavilyInput,
    #     CallbackManagerForToolRun,
    #     AsyncCallbackManagerForToolRun,
    # )
    # from langchain_core.pydantic_v1 import BaseModel, Field

    # from langchain_core.callbacks import (
    # AsyncCallbackManagerForToolRun,
    #     CallbackManagerForToolRun,
    # )
    # from langchain_core.pydantic_v1 import BaseModel, Field
    # from langchain_core.tools import BaseTool

    # ===
    from langchain_core.callbacks import (
        AsyncCallbackManagerForToolRun,
        CallbackManagerForToolRun,
    )
    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain_core.tools import BaseTool

    from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
    # from langchain_community.tools.tavily_search import (
    #     TavilySearchResults,
    # )

    LANGCHAIN_AVAILABLE = True

except Exception as e:
    print(f'{str(e)}\nNeed to install langchain: `{langchain_install_guide}`')

    LANGCHAIN_AVAILABLE = False


import logging
import importlib

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful, and honest assistant."""


class AnyEnginePipeline(BaseLLM):
    engine: Any  #: :meta private:
    # model_id: str = DEFAULT_MODEL_ID
    """Model name to use."""
    model_kwargs: Optional[dict] = None
    """Keyword arguments passed to the model."""
    pipeline_kwargs: Optional[dict] = None
    """Keyword arguments passed to the pipeline."""
    batch_size: int = 1
    """Batch size to use when passing multiple documents to generate."""
    streaming: bool = False

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def from_engine(
        cls,
        engine: Any,
        model_kwargs: Optional[dict] = None,
        **kwargs
    ):
        return cls(engine=engine, model_kwargs=model_kwargs, **kwargs)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            # "model_id": self.model_id,
            "model_kwargs": self.model_kwargs,
            # "pipeline_kwargs": self.pipeline_kwargs,
        }

    @property
    def _llm_type(self) -> str:
        return "engine_pipeline"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        # List to hold all results
        text_generations: List[str] = []
        stop_strings = stop
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]
            responses = []
            for p in batch_prompts:
                output = self.engine.generate_yield_string_final(p, stop_strings=stop_strings, **kwargs)
                responses.append(output[0])
            for j, (prompt, response) in enumerate(zip(batch_prompts, responses)):
                text = response
                if text.startswith(prompt):
                    text = text[len(prompt):]
                if stop is not None and any(x in text for x in stop):
                    text = text[:text.index(stop[0])]
                # print(f">>{text}")
                text_generations.append(text)
        return LLMResult(
            generations=[[Generation(text=text)] for text in text_generations]
        )


class ChatAnyEnginePipeline(BaseChatModel):
    """
    Wrapper for engine
    """
    llm: AnyEnginePipeline
    """LLM, must be of type HuggingFaceTextGenInference, HuggingFaceEndpoint, or 
        HuggingFaceHub."""
    system_message: SystemMessage = SystemMessage(content=DEFAULT_SYSTEM_PROMPT)
    tokenizer: Any = None
    model_id: Optional[str] = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.tokenizer = self.llm.engine.tokenizer
    
    @root_validator()
    def validate_llm(cls, values: dict) -> dict:
        # if not isinstance(
        #     values["llm"],
        #     (HuggingFaceTextGenInference, HuggingFaceEndpoint, HuggingFaceHub),
        # ):
        #     raise TypeError(
        #         "Expected llm to be one of HuggingFaceTextGenInference, "
        #         f"HuggingFaceEndpoint, HuggingFaceHub, received {type(values['llm'])}"
        #     )
        return values
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm_input = self._to_chat_prompt(messages)
        llm_result = self.llm._generate(
            prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
        )
        return self._to_chat_result(llm_result)
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm_input = self._to_chat_prompt(messages)
        llm_result = await self.llm._agenerate(
            prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
        )
        return self._to_chat_result(llm_result)

    def _to_chat_prompt(
        self,
        messages: List[BaseMessage],
    ) -> str:
        """Convert a list of messages into a prompt format expected by wrapped LLM."""
        if not messages:
            raise ValueError("At least one HumanMessage must be provided!")

        if not isinstance(messages[-1], HumanMessage):
            raise ValueError("Last message must be a HumanMessage!")

        messages_dicts = [self._to_chatml_format(m) for m in messages]

        return self.tokenizer.apply_chat_template(
            messages_dicts, tokenize=False, add_generation_prompt=True
        )

    def _to_chatml_format(self, message: BaseMessage) -> dict:
        """Convert LangChain message to ChatML format."""

        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

        return {"role": role, "content": message.content}

    @staticmethod
    def _to_chat_result(llm_result: LLMResult) -> ChatResult:
        chat_generations = []

        for g in llm_result.generations[0]:
            chat_generation = ChatGeneration(
                message=AIMessage(content=g.text), generation_info=g.generation_info
            )
            chat_generations.append(chat_generation)

        return ChatResult(
            generations=chat_generations, llm_output=llm_result.llm_output
        )
    
    def _resolve_model_id(self) -> None:
        self.model_id = "debug"

    @property
    def _llm_type(self) -> str:
        return "engine-chat-wrapper"




class TavilyInput(BaseModel):
    """Input for the Tavily tool."""

    query: str = Field(description="search query to look up")


class NewTavilySearchAPIWrapper(TavilySearchAPIWrapper):
    def clean_results(self, results: List[Dict]) -> List[Dict]:
        """Clean results from Tavily Search API."""
        clean_results = []
        for result in results:
            clean_results.append(
                {
                    "url": result["url"],
                    "content": result.get("raw_content", result["content"]),
                }
            )
        return clean_results


class NewTavilySearchResults(BaseTool):
    """Tool that queries the Tavily Search API and gets back json."""

    name: str = "tavily_search_results_json"
    description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    api_wrapper: NewTavilySearchAPIWrapper = Field(default_factory=NewTavilySearchAPIWrapper)
    max_results: int = 5
    args_schema: Type[BaseModel] = TavilyInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        """Use the tool."""
        try:
            return self.api_wrapper.results(
                query,
                self.max_results,
                include_answer=True,
                include_raw_content=True,
            )
        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        """Use the tool asynchronously."""
        try:
            return await self.api_wrapper.results_async(
                query,
                self.max_results,
                include_answer=True,
                include_raw_content=True,
            )
        except Exception as e:
            return repr(e)


FINAL_ANSWER_ACTION = "Final Answer:"
class LooseReActJsonSingleInputOutputParser(ReActJsonSingleInputOutputParser):
    def parse(self, text: str) -> AgentAction | AgentFinish:
        try:
            return super().parse(text)
        except OutputParserException as e:
            output = text
            if FINAL_ANSWER_ACTION in text:
                output = text.split(FINAL_ANSWER_ACTION)[-1].strip()
            return AgentFinish({"output": text}, text)




# web_search_system_prompt = """You are a helpful assistant. Please answer the following questions as best you can. You have access to the following tools:

# ## List of tools and their descriptions:
# {tools}

# ## Instructions
# The way you use the tools is by specifying a json blob.
# Specifically, this json should have an `action` key (with the name of the tool to use as specified above) and an `action_input` key (with the input to the tool with the corresponding required format as specified above).

# The only values that should be in the "action" field are: {tool_names}

# The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

# ```
# {{
#     "action": $TOOL_NAME,
#     "action_input": $INPUT
# }}
# ```

# ALWAYS use the following format:

# Question: the input question you must answer
# Thought: you should always think about what to do in the first step
# Action:
# ```
# $JSON_BLOB for action 1
# ```
# Observation: the result of the action you just performed
# Thought: you continue to think about what to do next
# Action:
# ```
# $JSON_BLOB for action 2 (if any)
# ```
# Observation: the result of the action
# ... (this Thought/Action/Observation can repeat N times)
# Thought: I now know the final answer
# Answer: `the final answer to the original input question`


# Begin! Below are a conversation between you and the user.
# """

# """
# {
#     "action": "tavily_search_results_json",
#     "action_input": {
#         "query": "langchain"
#     }
# }
# """


# [{'type': 'function', 'function': {'name': 'tavily_search_results_json', 'description': 'A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. Input should be a search query.', 'parameters': {'type': 'object', 'properties': {'query': {'description': 'search query to look up', 'type': 'string'}}, 'required': ['query']}}}]



web_search_system_prompt = """You are a helpful, intelligent and respectful assistant with access to the Internet via the `tavily_search_results_json` search engine tool. \
You provide answers and responses as accurately as possible to the user queries and questions, using the tools available to you. \
You may use your own knowledge to reply to the user. However, if you are not confident about your knowledge, or you do not have the up-to-date knowledge and abilitiy to answer the questions, please use the search tool to query appropriately.

You understand that you have to craft an informative and search-engine-friendly query given the user's question for the engine to retrieve the most relevant information. \
You also understand that if the question is complex, you may need to reason your thoughts step by step, and may call the search engine multiple times if needed. However, you must use the least API call as possible!
If you have used the search engine, you should include in your final response citations of the website links you have retrieved.

To use the search engine, you must first speak out your thought, then follow by an action as a json blob and understand the observation, and produce the final answer. ALWAYS use the following format:

Question: the input user question you must answer
Thought: you should always think about what to do in the first step
Action:
```
{{
    "action": "tavily_search_results_json",
    "action_input": {{
        "query": "search query 1"
    }}
}}
```
Observation: the result of the search query 1 you just performed
Thought: you continue to think about what to query next, if necessary
Action:
```
{{
    "action": "tavily_search_results_json",
    "action_input": {{
        "query": "search query 1"
    }}
}}
```
Observation: the result of the search query 2 you just performed
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final answer: the final answer to the original user's input question
Citation: ...

You are provided the following concrete examples, please study them and understand your task.

### Example 1

Question: Who is the wife of the current US president?
Thought: This question is twofold and a single search query may not suffice. First I need to find out who is the current US president, then I need to find out who his wife is.
Action:
```
{{
    "action": "tavily_search_results_json",
    "action_input": {{
        "query": "Current US president"
    }}
}}
```
Observation: [{{'url': 'https://en.wikipedia.org/wiki/Joe_Biden', 'content': 'Joe Biden is the current US president. He is the 46th US president.'}}]
Thought: Now I need to find out who is the wife of Joe Biden
Action:
```
{{
    "action": "tavily_search_results_json",
    "action_input": {{
        "query": "Who is the wife of Joe Biden?"
    }}
}}
```
Observation: [{{'url': 'https://en.wikipedia.org/wiki/Jill_Biden', 'content': 'The wife of Joe Biden is Jill Biden, who is an American educator.'}}]
Thought: I now know the final answer
Final answer: The wife of the current US president is Jill Biden.
Citation: 
* https://en.wikipedia.org/wiki/Joe_Biden
* https://en.wikipedia.org/wiki/Jill_Biden

### Example 2

Question: What is langchain?
Thought: I think I should query the internet to understand what is langchain
Action:
```
{{
    "action": "tavily_search_results_json",
    "action_input": {{
        "query": "what is langchain?"
    }}
}}
```
Observation: [{{'url': 'https://python.langchain.com/docs/get_started/introduction/', 'content': 'LangChain is a framework for developing applications powered by large language models (LLMs).'}}]
Thought: I now know the final answer
Final answer: From my search query, Langchain is a framework for building applications using Large Language Models or LLMs.
Citation: 
* https://python.langchain.com/docs/get_started/introduction/


Let's begin! Below is the question from the user.
"""
# FINAL REMARKS: The user may not speak English and may ask you questions in any language. Thus, while your Thought, Action and Observation is in English, your `Final answer` should be in the same language as the user's query.


"""


"""



def create_web_search_engine():
    from ..globals import MODEL_ENGINE
    # from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_core.utils.function_calling import (
        convert_to_openai_function,
        convert_to_openai_tool,
    )
    from langchain_core.exceptions import OutputParserException
    from langchain_core.agents import AgentAction, AgentFinish
    web_search_llm = AnyEnginePipeline.from_engine(MODEL_ENGINE)
    web_search_chat_model = ChatAnyEnginePipeline(llm=web_search_llm)
    if "TAVILY_API_KEY" not in os.environ:
        raise ValueError(f'TAVILY_API_KEY is not found to use websearch, please `export TAVILY_API_KEY=YOUR_TAVILY_API_KEY`')

    tools = [NewTavilySearchResults(max_results=1)]
    formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
    # tools = load_tools(["llm-math"], llm=web_search_llm)
    # formatted_tools = render_text_description_and_args(tools)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            # (
            #     "system",
            #     web_search_system_prompt,
            # ),
            (
                "human",
                web_search_system_prompt + "\n{input}\n{agent_scratchpad}"
                # "{input}\n\n{agent_scratchpad}"
            )
        ]
    )
    prompt = prompt_template.partial(
        tools=formatted_tools,
        tool_names=", ".join([t.name for t in tools]),
    )
    chat_model_with_stop = web_search_chat_model.bind(stop=["\nObservation"])
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        }
        | prompt
        | chat_model_with_stop
        | LooseReActJsonSingleInputOutputParser()
    )
        # | ReActJsonSingleInputOutputParser()

    # instantiate AgentExecutor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    # agent_executor.invoke({"input": "What is langchain?"})
    return web_search_llm, web_search_chat_model, agent_executor





# if LANGCHAIN_AVAILABLE:
#     class LooseReActJsonSingleInputOutputParser(ReActJsonSingleInputOutputParser):
#         def parse(self, text: str) -> AgentAction | AgentFinish:
#             try:
#                 return super().parse(text)
#             except OutputParserException as e:
#                 return AgentFinish({"output": text}, text)


    # class ChatHuggingfaceFromLocalPipeline(ChatHuggingFace):
    #     @root_validator()
    #     def validate_llm(cls, values: dict) -> dict:
    #         return values
    #     def _resolve_model_id(self) -> None:
    #         """Resolve the model_id from the LLM's inference_server_url"""
    #         self.model_id = self.llm.model_id


    # class NewHuggingfacePipeline(HuggingFacePipeline):
    #     bos_token = "<bos>"
    #     add_bos_token = True

    #     @classmethod
    #     def from_model_id(
    #         cls,
    #         model_id: str,
    #         task: str,
    #         backend: str = "default",
    #         device: Optional[int] = -1,
    #         device_map: Optional[str] = None,
    #         model_kwargs: Optional[dict] = None,
    #         pipeline_kwargs: Optional[dict] = None,
    #         batch_size: int = 2,
    #         model = None,
    #         **kwargs: Any,
    #     ) -> HuggingFacePipeline:
    #         """Construct the pipeline object from model_id and task."""
    #         try:
    #             from transformers import (
    #                 AutoModelForCausalLM,
    #                 AutoModelForSeq2SeqLM,
    #                 AutoTokenizer,
    #             )
    #             from transformers import pipeline as hf_pipeline

    #         except ImportError:
    #             raise ValueError(
    #                 "Could not import transformers python package. "
    #                 "Please install it with `pip install transformers`."
    #             )

    #         _model_kwargs = model_kwargs or {}
    #         tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)
    #         if model is None:
    #             try:
    #                 if task == "text-generation":
    #                     if backend == "openvino":
    #                         try:
    #                             from optimum.intel.openvino import OVModelForCausalLM

    #                         except ImportError:
    #                             raise ValueError(
    #                                 "Could not import optimum-intel python package. "
    #                                 "Please install it with: "
    #                                 "pip install 'optimum[openvino,nncf]' "
    #                             )
    #                         try:
    #                             # use local model
    #                             model = OVModelForCausalLM.from_pretrained(
    #                                 model_id, **_model_kwargs
    #                             )

    #                         except Exception:
    #                             # use remote model
    #                             model = OVModelForCausalLM.from_pretrained(
    #                                 model_id, export=True, **_model_kwargs
    #                             )
    #                     else:
    #                         model = AutoModelForCausalLM.from_pretrained(
    #                             model_id, **_model_kwargs
    #                         )
    #                 elif task in ("text2text-generation", "summarization", "translation"):
    #                     if backend == "openvino":
    #                         try:
    #                             from optimum.intel.openvino import OVModelForSeq2SeqLM

    #                         except ImportError:
    #                             raise ValueError(
    #                                 "Could not import optimum-intel python package. "
    #                                 "Please install it with: "
    #                                 "pip install 'optimum[openvino,nncf]' "
    #                             )
    #                         try:
    #                             # use local model
    #                             model = OVModelForSeq2SeqLM.from_pretrained(
    #                                 model_id, **_model_kwargs
    #                             )

    #                         except Exception:
    #                             # use remote model
    #                             model = OVModelForSeq2SeqLM.from_pretrained(
    #                                 model_id, export=True, **_model_kwargs
    #                             )
    #                     else:
    #                         model = AutoModelForSeq2SeqLM.from_pretrained(
    #                             model_id, **_model_kwargs
    #                         )
    #                 else:
    #                     raise ValueError(
    #                         f"Got invalid task {task}, "
    #                         f"currently only {VALID_TASKS} are supported"
    #                     )
    #             except ImportError as e:
    #                 raise ValueError(
    #                     f"Could not load the {task} model due to missing dependencies."
    #                 ) from e
    #         else:
    #             print(f'PIpeline skipping creation of model because model is given')

    #         if tokenizer.pad_token is None:
    #             tokenizer.pad_token_id = model.config.eos_token_id

    #         if (
    #             (
    #                 getattr(model, "is_loaded_in_4bit", False)
    #                 or getattr(model, "is_loaded_in_8bit", False)
    #             )
    #             and device is not None
    #             and backend == "default"
    #         ):
    #             logger.warning(
    #                 f"Setting the `device` argument to None from {device} to avoid "
    #                 "the error caused by attempting to move the model that was already "
    #                 "loaded on the GPU using the Accelerate module to the same or "
    #                 "another device."
    #             )
    #             device = None

    #         if (
    #             device is not None
    #             and importlib.util.find_spec("torch") is not None
    #             and backend == "default"
    #         ):
    #             import torch

    #             cuda_device_count = torch.cuda.device_count()
    #             if device < -1 or (device >= cuda_device_count):
    #                 raise ValueError(
    #                     f"Got device=={device}, "
    #                     f"device is required to be within [-1, {cuda_device_count})"
    #                 )
    #             if device_map is not None and device < 0:
    #                 device = None
    #             if device is not None and device < 0 and cuda_device_count > 0:
    #                 logger.warning(
    #                     "Device has %d GPUs available. "
    #                     "Provide device={deviceId} to `from_model_id` to use available"
    #                     "GPUs for execution. deviceId is -1 (default) for CPU and "
    #                     "can be a positive integer associated with CUDA device id.",
    #                     cuda_device_count,
    #                 )
    #         if device is not None and device_map is not None and backend == "openvino":
    #             logger.warning("Please set device for OpenVINO through: " "'model_kwargs'")
    #         if "trust_remote_code" in _model_kwargs:
    #             _model_kwargs = {
    #                 k: v for k, v in _model_kwargs.items() if k != "trust_remote_code"
    #             }
    #         _pipeline_kwargs = pipeline_kwargs or {}
    #         pipeline = hf_pipeline(
    #             task=task,
    #             model=model,
    #             tokenizer=tokenizer,
    #             device=device,
    #             device_map=device_map,
    #             batch_size=batch_size,
    #             model_kwargs=_model_kwargs,
    #             **_pipeline_kwargs,
    #         )
    #         if pipeline.task not in VALID_TASKS:
    #             raise ValueError(
    #                 f"Got invalid task {pipeline.task}, "
    #                 f"currently only {VALID_TASKS} are supported"
    #             )
    #         return cls(
    #             pipeline=pipeline,
    #             model_id=model_id,
    #             model_kwargs=_model_kwargs,
    #             pipeline_kwargs=_pipeline_kwargs,
    #             batch_size=batch_size,
    #             **kwargs,
    #         )
        
    #     def _generate(
    #         self,
    #         prompts: List[str],
    #         stop: Optional[List[str]] = None,
    #         run_manager: Optional[CallbackManagerForLLMRun] = None,
    #         **kwargs: Any,
    #     ) -> LLMResult:
    #         # List to hold all results
    #         text_generations: List[str] = []
    #         pipeline_kwargs = kwargs.get("pipeline_kwargs", self.pipeline_kwargs)
    #         pipeline_kwargs = pipeline_kwargs if len(pipeline_kwargs) > 0 else self.pipeline_kwargs
    #         for i in range(0, len(prompts), self.batch_size):
    #             batch_prompts = prompts[i : i + self.batch_size]
    #             bos_token = self.pipeline.tokenizer.convert_ids_to_tokens(self.pipeline.tokenizer.bos_token_id)
    #             for i in range(len(batch_prompts)):
    #                 if not batch_prompts[i].startswith(bos_token) and self.add_bos_token:
    #                     batch_prompts[i] = bos_token + batch_prompts[i]
    #             # print(f'PROMPT: {stop=} {pipeline_kwargs=} ==================\n{batch_prompts[0]}\n==========================')
    #             # Process batch of prompts
    #             responses = self.pipeline(
    #                 batch_prompts,
    #                 **pipeline_kwargs,
    #             )
    #             # Process each response in the batch
    #             for j, (prompt, response) in enumerate(zip(batch_prompts, responses)):
    #                 if isinstance(response, list):
    #                     # if model returns multiple generations, pick the top one
    #                     response = response[0]
    #                 if self.pipeline.task == "text-generation":
    #                     text = response["generated_text"]
    #                 elif self.pipeline.task == "text2text-generation":
    #                     text = response["generated_text"]
    #                 elif self.pipeline.task == "summarization":
    #                     text = response["summary_text"]
    #                 elif self.pipeline.task in "translation":
    #                     text = response["translation_text"]
    #                 else:
    #                     raise ValueError(
    #                         f"Got invalid task {self.pipeline.task}, "
    #                         f"currently only {VALID_TASKS} are supported"
    #                     )
    #                 # Append the processed text to results
    #                 if text.startswith(prompt):
    #                     text = text[len(prompt):]
    #                 if stop is not None and any(x in text for x in stop):
    #                     text = text[:text.index(stop[0])]
    #                 # print(f">>{text}")
    #                 text_generations.append(text)
    #         return LLMResult(
    #             generations=[[Generation(text=text)] for text in text_generations]
    #         )

