import logging
from typing import Dict, Optional

from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.utils import get_from_dict_or_env
from pydantic import root_validator

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


if __name__ == "__main__":
    from langchain.schema import HumanMessage
    # from langchain.callbacks.base import CallbackManager
    # from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

    chat = SelfHostedChatOpenAI(
        model_name="chatglm-6b",
        openai_api_base="http://192.168.0.53/v1",
        openai_api_key="xxx",
        max_tokens=100,
        # streaming=True,
        # verbose=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

    print(chat([HumanMessage(content="你好")]))