"""
A text generation model with stream decoding.
"""
import copy

import torch
from tenacity import retry, stop_after_attempt, wait_fixed
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BertTokenizerFast,
    LogitsProcessorList,
    MinNewTokensLengthLogitsProcessor,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
)

from . import MODEL
from .choice import map_choice
from .tokenizer import StreamTokenizer


class StreamModel:
    """StreamModel wraps around a language model to provide stream decoding."""

    def __init__(
        self,
        model,
        tokenizer,
        min_tokens=0,
        max_tokens=2048,
        temperature=1.0,
        top_p=1.0,
        n=1,
        logprobs=0,
        echo=False,
        remove_whitespace=True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

        self.generate_kwargs = {
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "logprobs": logprobs,
            "echo": echo,
            "remove_whitespace": remove_whitespace,
        }
        self.generate_kwargs.update(kwargs)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def __call__(self, prompt=None, messages=None, **kwargs):
        """Create a completion stream for the provided prompt."""
        if messages is not None:
            prompt = self.chat_process(messages)

        if isinstance(prompt, str):
            input_ids = self.tokenize(prompt)
        elif isinstance(prompt, torch.Tensor) and prompt.dim() == 1:
            input_ids = prompt
        else:
            raise TypeError("prompt must be a string or a 1-d tensor")

        self.generate_kwargs.update(kwargs)

        # Ensure arguments are non-negative.
        n = max(self.generate_kwargs.get("n"), 1)
        min_tokens = max(self.generate_kwargs.get("min_tokens"), 0)
        max_tokens = max(self.generate_kwargs.get("max_tokens"), 1)
        logprobs = max(self.generate_kwargs.get("logprobs"), 0)

        # Keep track of the finish reason of each sequence.
        finish_reasons = [None] * n

        # Create stateful detokenizer for each sequence.
        detokenizers = []
        for i in range(n):
            detokenizers.append(
                StreamTokenizer(self.tokenizer, remove_whitespace=self.generate_kwargs.get("remove_whitespace"))
            )

        # Echo prompt tokens if required.
        if self.generate_kwargs.get("echo"):
            for token in input_ids:
                samples = self._sample(token, 0, [], []) if logprobs > 0 else {}
                for i in range(n):
                    text = detokenizers[i].decode(token)
                    offset = detokenizers[i].start
                    yield map_choice(text, i, text_offset=offset, **samples)

        # Generate completion tokens.
        for (
            tokens,
            token_logprobs,
            top_tokens,
            top_logprobs,
            status,
        ) in self.generate(
            input_ids[None, :].repeat(n, 1),
            logprobs=logprobs,
            min_new_tokens=min_tokens,
            max_new_tokens=max_tokens,
            temperature=self.generate_kwargs.get("temperature"),
            top_p=self.generate_kwargs.get("top_p"),
        ):
            for i in range(n):
                # Check and update the finish status of the sequence.
                if finish_reasons[i]:
                    continue
                if status[i] == 0:
                    finish_reasons[i] = "stop"
                elif status[i] == -1:
                    finish_reasons[i] = "length"

                # Collect samples of the most likely tokens if required.
                samples = (
                    self._sample(
                        token=tokens[i],
                        token_logprob=token_logprobs[i],
                        top_tokens=top_tokens[i],
                        top_logprobs=top_logprobs[i],
                    )
                    if logprobs > 0
                    else {}
                )

                # Yield predicted tokens.
                text = detokenizers[i].decode(tokens[i])
                offset = detokenizers[i].start
                yield map_choice(
                    text,
                    i,
                    text_offset=offset,
                    finish_reason=finish_reasons[i],
                    **samples,
                )

    def chatglm_chat(self, messages=None, **kwargs):
        max_tokens = kwargs.get("max_tokens", 2048)
        history = [x["content"] for x in messages]
        query = history.pop()
        assert (
            len(history) % 2 == 0
        ), f"History should be even length. current history is: {history}"
        history = [[history[i], history[i + 1]] for i in range(0, len(history), 2)]

        for response, updates in self.model.stream_chat(
            self.tokenizer,
            query,
            history,
            max_length=max_tokens,
            top_p=kwargs.get("top_p", 0.7),
            temperature=kwargs.get("temperature", 0.95),
        ):
            yield map_choice(
                response,
                0,
                chat=True,
            )

    def chat_process(self, messages):
        if len(messages) == 1:
            prompt = messages[0]["content"]
        else:
            prompt = ""
            for i, message in enumerate(messages):
                if "chatglm" in MODEL:
                    if i % 2 == 0:
                        prompt += f"[Round {i//2}]\n问：{message['content']}\n"
                    else:
                        prompt += f"答：{message['content']}\n"
                else:
                    prompt += f"{message['role']}：{message['content']}\n"

        return prompt

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
    def _infer(self, model_fn, **kwargs):
        """Call a model function in inference mode with auto retrying."""
        # This is a temporary workaround for bitsandbytes #162:
        # https://github.com/TimDettmers/bitsandbytes/issues/162
        with torch.inference_mode():
            return model_fn(**kwargs)

    def _sample(self, token, token_logprob, top_tokens, top_logprobs):
        """Sample log probabilities of the most likely tokens."""
        token = self.tokenizer.decode(token)
        top_tokens = self.tokenizer.batch_decode(top_tokens)

        # Do not use tensor operations as arguments may be of list type.
        token_logprob = round(float(token_logprob), 8)
        top_logprobs = [round(float(p), 8) for p in top_logprobs]

        # Always include the log probability of the selected token.
        top_logprobs = dict(zip(top_tokens, top_logprobs))
        top_logprobs[token] = token_logprob

        return {
            "token": token,
            "token_logprob": token_logprob,
            "top_logprobs": top_logprobs,
        }

    def _logits_processor(self, config, input_length):
        """Set up logits processor based on the generation config."""
        processor = LogitsProcessorList()

        # Add processor for enforcing a min-length of new tokens.
        if (
            config.min_new_tokens is not None
            and config.min_new_tokens > 0
            and config.eos_token_id is not None
        ):
            processor.append(
                MinNewTokensLengthLogitsProcessor(
                    prompt_length_to_skip=input_length,
                    min_new_tokens=config.min_new_tokens,
                    eos_token_id=config.eos_token_id,
                )
            )

        # Add processor for scaling output probability distribution.
        if (
                config.temperature is not None
                and config.temperature > 0
                and config.temperature != 1.0
        ):
            processor.append(TemperatureLogitsWarper(config.temperature))

        # Add processor for nucleus sampling.
        if config.top_p is not None and 0 < config.top_p < 1:
            processor.append(TopPLogitsWarper(config.top_p))

        return processor

    def tokenize(self, text):
        """Tokenize a string into a tensor of token IDs."""
        if self.tokenizer.__str__().startswith("BertTokenizer"):
            batch = self.tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)
        else:
            batch = self.tokenizer.encode(text, return_tensors="pt")
        return batch[0].to(self.device)

    def generate(self, input_ids, logprobs=0, **kwargs):
        """Generate a stream of predicted tokens using the language model."""

        # Store the original batch size and input length.
        batch_size = input_ids.shape[0]
        input_length = input_ids.shape[-1]

        # Separate model arguments from generation config.
        config = self.model.generation_config
        if self.tokenizer.__str__().startswith("BertTokenizer"):
            config.bos_token_id = self.tokenizer.cls_token_id
            config.eos_token_id = self.tokenizer.sep_token_id
            config.pad_token_id = self.tokenizer.pad_token_id
        config = copy.deepcopy(config)

        kwargs = config.update(**kwargs)
        kwargs["output_attentions"] = False
        kwargs["output_hidden_states"] = False
        kwargs["use_cache"] = config.use_cache

        # Collect special token IDs.
        pad_token_id = config.pad_token_id
        bos_token_id = config.bos_token_id
        eos_token_id = config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id[0]

        # Generate from eos if no input is specified.
        if input_length == 0:
            input_ids = input_ids.new_ones((batch_size, 1)).long()
            if eos_token_id is not None:
                input_ids = input_ids * eos_token_id[0]
            input_length = 1

        # Prepare inputs for encoder-decoder checkpoints.
        if self.model.config.is_encoder_decoder:
            # Get outputs from the encoder.
            encoder = self.model.get_encoder()
            encoder_kwargs = kwargs.copy()
            encoder_kwargs.pop("use_cache", None)
            encoder_kwargs["input_ids"] = input_ids
            encoder_kwargs["return_dict"] = True
            encoder_outputs = self._infer(encoder, **encoder_kwargs)
            kwargs["encoder_outputs"] = encoder_outputs

            # Reinitialize inputs for the decoder.
            decoder_start_token_id = config.decoder_start_token_id
            if decoder_start_token_id is None:
                decoder_start_token_id = bos_token_id
            input_ids = input_ids.new_ones((batch_size, 1))
            input_ids = input_ids * decoder_start_token_id
            input_length = 1

        # Set up logits processor.
        processor = self._logits_processor(config, input_length)

        # Keep track of which sequences are already finished.
        unfinished = input_ids.new_ones(batch_size)

        # Start auto-regressive generation.
        while True:
            inputs = self.model.prepare_inputs_for_generation(
                input_ids, **kwargs
            )  # noqa: E501
            outputs = self._infer(
                self.model,
                **inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            # Pre-process the probability distribution of the next tokens.
            logits = outputs.logits[:, -1, :]
            with torch.inference_mode():
                logits = processor(input_ids, logits)
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Select deterministic or stochastic decoding strategy.
            if (config.top_p is not None and config.top_p <= 0) or (
                    config.temperature is not None and config.temperature <= 0
            ):
                tokens = torch.argmax(probs, dim=-1)[:, None]
            else:
                tokens = torch.multinomial(probs, num_samples=1)

            # Collect log probabilities of the selected tokens.
            token_logprobs = torch.gather(probs, 1, tokens)
            token_logprobs = torch.log(token_logprobs + 1e-7).squeeze(1)
            tokens = tokens.squeeze(1)

            # Collect log probabilities of the most likely tokens.
            top_logprobs, top_tokens = probs.topk(logprobs)
            top_logprobs = torch.log(top_logprobs + 1e-7)

            # Finished sequences should have their next token be a padding.
            if pad_token_id is not None:
                tokens = tokens * unfinished + pad_token_id * (1 - unfinished)

            # Append selected tokens to the inputs.
            input_ids = torch.cat([input_ids, tokens[:, None]], dim=-1)

            # Extract past key values from model outputs.
            if "past_key_values" in outputs:
                kwargs["past_key_values"] = outputs.past_key_values
            elif "mems" in outputs:
                kwargs["past_key_values"] = outputs.mems
            elif "past_buckets_states" in outputs:
                kwargs["past_key_values"] = outputs.past_buckets_states

            # Mark sequences with eos tokens as finished.
            if eos_token_id is not None:
                not_eos = sum(tokens != i for i in eos_token_id)
                unfinished = unfinished.mul(not_eos.long())

            # Set status to -1 if exceeded the max length.
            status = unfinished.clone()
            if input_ids.shape[-1] - input_length >= config.max_new_tokens:
                status = 0 - status

            # Yield predictions and status.
            yield tokens, token_logprobs, top_tokens, top_logprobs, status

            # Stop when finished or exceeded the max length.
            if status.max() <= 0:
                break


def load_model(
    name_or_path,
    revision=None,
    cache_dir=None,
    load_in_8bit=False,
    local_files_only=False,
    trust_remote_code=False,
    half_precision=False,
    tokenizer_name=None,
    remove_whitespace=True,
):
    """Load a text generation model and make it stream-able."""
    kwargs = {
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    if revision:
        kwargs["revision"] = revision
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    if tokenizer_name is not None:
        if tokenizer_name.lower() == "bert":
            tokenizer_cls = BertTokenizerFast
        else:
            tokenizer_cls = AutoTokenizer
    else:
        tokenizer_cls = AutoTokenizer

    tokenizer = tokenizer_cls.from_pretrained(name_or_path, **kwargs)

    # Set device mapping and quantization options if CUDA is available.
    if torch.cuda.is_available():
        kwargs = kwargs.copy()
        kwargs["device_map"] = "auto"
        kwargs["load_in_8bit"] = load_in_8bit

        # Override the dtype to float16 as required by bitsandbytes.
        if half_precision or load_in_8bit:
            kwargs["torch_dtype"] = torch.float16

    # Support both decoder-only and encoder-decoder checkpoints.
    try:
        if "chatglm" in name_or_path:
            model = AutoModel.from_pretrained(name_or_path, trust_remote_code=True).half()
        else:
            model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
    except ValueError:
        model = AutoModelForSeq2SeqLM.from_pretrained(name_or_path, **kwargs)

    # Check if the model has text generation capabilities.
    if not model.can_generate():
        raise TypeError(f"{name_or_path} is not a text generation model")

    return StreamModel(model, tokenizer, remove_whitespace=remove_whitespace)
