import importlib.util
import logging
import pickle
from typing import Any, Callable, List, Mapping, Optional

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from pydantic import BaseModel, Extra
from transformers import GPT2LMHeadModel, BertTokenizer

from basaran.choice import reduce_choice
from basaran.model import StreamModel

logger = logging.getLogger()


def _generate_text(
    pipeline,
    prompt: str,
    *args: Any,
    stop: Optional[List[str]] = None,
    **kwargs: Any,
) -> str:
    buffers = {}
    for choice in pipeline(prompt, **kwargs):
        index = choice["index"]
        if index not in buffers:
            buffers[index] = []
        buffers[index].append(choice)

    choices = []
    for _, buffer in buffers.items():
        if buffer:
            choices.append(reduce_choice(buffer))

    text = choices[0]["text"]

    if stop is not None:
        text = enforce_stop_tokens(text, stop)

    return prompt + text.replace("[UNK]", "")


def _send_pipeline_to_device(pipeline: Any, device: int) -> Any:
    """Send a pipeline to a device on the cluster."""
    if isinstance(pipeline, str):
        with open(pipeline, "rb") as f:
            pipeline = pickle.load(f)

    if importlib.util.find_spec("torch") is not None:
        import torch

        if not torch.cuda.is_available() or device == -1:
            device = "cpu"

        pipeline.device = torch.device(device)
        pipeline.model = pipeline.model.to(pipeline.device)

    return pipeline


class SelfHostedPipeline(LLM, BaseModel):
    pipeline_ref: Any
    client: Any
    inference_fn: Callable = _generate_text
    model_load_fn: Callable
    load_fn_kwargs: Optional[dict] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        _load_fn_kwargs = self.load_fn_kwargs or {}
        self.pipeline_ref = self.model_load_fn(**_load_fn_kwargs)
        self.client = self.inference_fn

    @classmethod
    def from_pipeline(
        cls,
        pipeline: Any,
        device: int = 0,
        **kwargs: Any,
    ) -> LLM:
        """Init the SelfHostedPipeline from a pipeline object or string."""
        if not isinstance(pipeline, str):
            logger.warning(
                "Serializing pipeline. "
                "Note, it can be quite slow"
                "to serialize and send large checkpoints with each execution. "
                "Consider sending the pipeline"
                "to the cluster and passing the path to the pipeline instead."
            )

        load_fn_kwargs = {"pipeline": pipeline, "device": device}
        return cls(
            load_fn_kwargs=load_fn_kwargs,
            model_load_fn=_send_pipeline_to_device,
            **kwargs,
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    @property
    def _llm_type(self) -> str:
        return "self_hosted_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self.client(pipeline=self.pipeline_ref, prompt=prompt, stop=stop)


def load_pipeline(model_name_or_path, device=-1, **kwargs):
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
    pipeline = StreamModel(model, tokenizer, **kwargs)
    return _send_pipeline_to_device(pipeline, device)


if __name__ == "__main__":
    llm = SelfHostedPipeline(
        model_load_fn=load_pipeline,
        load_fn_kwargs={
            "model_name_or_path": "../checkpoints/gpt2-abstract/checkpoint-500000",
            "device": -1,
            "remove_whitespace": True,
            "max_tokens": 256,
        },
    )

    print(llm._generate(["为了回收某石英脉型金矿中的金,在粗磨条件下进行了尼尔森选矿机选矿试验研究。"]))
