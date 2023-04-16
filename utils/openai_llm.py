import logging
import sys
from typing import Optional, Dict, Any, List

from colorama import init, Fore
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)
from langchain.utils import get_from_dict_or_env
from pydantic import root_validator

init(autoreset=True)
logger = logging.getLogger()


class SelfHostedOpenAI(OpenAI):

    openai_api_base: Optional[str] = None

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_api_key = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        openai_api_base = get_from_dict_or_env(
            values, "openai_api_base", "OPENAI_API_BASE"
        )
        try:
            import openai

            openai.api_base = openai_api_base
            openai.api_key = openai_api_key
            values["client"] = openai.Completion
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please it install it with `pip install openai`."
            )
        if values["streaming"] and values["n"] > 1:
            raise ValueError("Cannot stream results when n > 1.")
        if values["streaming"] and values["best_of"] > 1:
            raise ValueError("Cannot stream results when best_of > 1.")
        return values


class SelfHostedChatOpenAI(ChatOpenAI):

    openai_api_base: Optional[str] = None

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_api_key = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        openai_api_base = get_from_dict_or_env(
            values, "openai_api_base", "OPENAI_API_BASE"
        )
        openai_organization = get_from_dict_or_env(
            values,
            "openai_organization",
            "OPENAI_ORGANIZATION",
            default="",
        )
        try:
            import openai

            openai.api_base = openai_api_base
            openai.api_key = openai_api_key
            if openai_organization:
                openai.organization = openai_organization
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        try:
            values["client"] = openai.ChatCompletion
        except AttributeError:
            raise ValueError(
                "`openai` has no `ChatCompletion` attribute, this is likely "
                "due to an old version of the openai package. Try upgrading it "
                "with `pip install --upgrade openai`."
            )
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["n"] > 1 and values["streaming"]:
            raise ValueError("n must be 1 when streaming.")
        return values


class StreamingStdOutCallbackHandlerWithColor(StreamingStdOutCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        sys.stdout.write(Fore.BLUE + token)
        sys.stdout.flush()


class ChatGLMConversationBufferWindowMemory(ConversationBufferWindowMemory):
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return history buffer."""

        if self.return_messages:
            buffer: Any = self.buffer[-self.k * 2:]
        else:
            buffer = self.get_buffer_string(
                self.buffer[-self.k * 2:],
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        return {self.memory_key: buffer}

    @staticmethod
    def get_buffer_string(
        messages: List[BaseMessage], human_prefix: str = "问", ai_prefix: str = "答"
    ) -> str:
        """Get buffer string of messages."""
        string_messages, i = [], 0
        for m in messages:
            if isinstance(m, HumanMessage):
                role = human_prefix
                string_messages.append(f"[Round {i}]\n{role}：{m.content}")
                i += 1
            elif isinstance(m, AIMessage):
                role = ai_prefix
                string_messages.append(f"{role}：{m.content}")
            else:
                raise ValueError(f"Got unsupported message type: {m}")

        return "\n".join(string_messages) + f"\n[Round {i}]"


class ChineseAlpacaConversationBufferWindowMemory(ChatGLMConversationBufferWindowMemory):
    @staticmethod
    def get_buffer_string(
        messages: List[BaseMessage], human_prefix: str = "### Instruction", ai_prefix: str = "### Response"
    ) -> str:
        """Get buffer string of messages."""
        string_messages = []
        for m in messages:
            if isinstance(m, HumanMessage):
                role = human_prefix
                string_messages.append(f"{role}:\n\n{m.content}")
            elif isinstance(m, AIMessage):
                role = ai_prefix
                string_messages.append(f"{role}:\n\n{m.content}")
            else:
                raise ValueError(f"Got unsupported message type: {m}")

        return "\n\n".join(string_messages)


class FireFlyConversationBufferWindowMemory(ChatGLMConversationBufferWindowMemory):
    @staticmethod
    def get_buffer_string(
        messages: List[BaseMessage], human_prefix: str = "Human", ai_prefix: str = "Assistant"
    ) -> str:
        """Get buffer string of messages."""
        string_messages = []
        for m in messages:
            if isinstance(m, HumanMessage):
                string_messages.append(f"<s>{m.content}</s>")
            elif isinstance(m, AIMessage):
                string_messages.append(f"</s>{m.content}</s>")
            else:
                raise ValueError(f"Got unsupported message type: {m}")

        return "".join(string_messages)


class PhoenixConversationBufferWindowMemory(ChatGLMConversationBufferWindowMemory):
    @staticmethod
    def get_buffer_string(
        messages: List[BaseMessage], human_prefix: str = "Human", ai_prefix: str = "Assistant"
    ) -> str:
        """Get buffer string of messages."""
        string_messages = []
        for m in messages:
            if isinstance(m, HumanMessage):
                role = human_prefix
            elif isinstance(m, AIMessage):
                role = ai_prefix
            else:
                raise ValueError(f"Got unsupported message type: {m}")
            string_messages.append(f"{role}: <s>{m.content}</s>")
        return "".join(string_messages)


def start_chat_by_chain(
    model_name, openai_api_base, max_tokens=2048, k=5, prompt=None, verbose=False,
):
    llm = SelfHostedChatOpenAI(
        model_name=model_name,
        openai_api_base=openai_api_base,
        openai_api_key="xxx",
        max_tokens=max_tokens,
        streaming=True,
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandlerWithColor()]),
    )

    if "chatglm" in model_name:
        memory = ChatGLMConversationBufferWindowMemory(
            k=k, human_prefix="问", ai_prefix="答"
        )
    elif model_name == "chinese-alpaca":
        memory = ChineseAlpacaConversationBufferWindowMemory(
            k=k, human_prefix="### Instruction", ai_prefix="### Response"
        )
    elif "firefly" in model_name:
        memory = FireFlyConversationBufferWindowMemory(
            k=k, human_prefix="Human", ai_prefix="Assistant"
        )
    elif "phoenix" in model_name:
        memory = PhoenixConversationBufferWindowMemory(
            k=k, human_prefix="Human", ai_prefix="Assistant"
        )
    else:
        raise ValueError(f"Got unsupported model name: {model_name}")

    chat_chain = ConversationChain(llm=llm, memory=memory, verbose=verbose)
    if prompt:
        chat_chain.prompt = prompt

    while True:
        user_input = input(Fore.BLUE + "HUMAN: ")
        print(Fore.BLUE + "AI: ")
        chat_chain.predict(input=user_input)
        print("\n")
