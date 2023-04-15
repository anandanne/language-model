import json
import logging
import os
from enum import Enum

import colorama
import tiktoken
import urllib3
from duckduckgo_search import ddg

from . import shared
from .presets import (
    MODEL_TOKEN_LIMIT,
    BILLING_NOT_APPLICABLE_MSG,
    DEFAULT_TOKEN_LIMIT,
    WEBSEARCH_PTOMPT_TEMPLATE,
    STANDARD_ERROR_MSG,
    NO_INPUT_MSG,
    NO_APIKEY_MSG,
    TOKEN_OFFSET,
    REDUCE_TOKEN_FACTOR,
    HISTORY_DIR,
)
from .utils import (
    construct_user,
    construct_assistant,
    count_token,
    add_source_numbers,
    replace_today,
    hide_middle_chars,
    save_file,
)


class ModelType(Enum):
    Unknown = -1
    OpenAI = 0
    ChatGLM = 1
    LLaMA = 2
    Bloom = 3

    @classmethod
    def get_type(cls, model_name: str):
        model_name_lower = model_name.lower()
        if "gpt" in model_name_lower:
            model_type = ModelType.OpenAI
        elif "chatglm" in model_name_lower:
            model_type = ModelType.ChatGLM
        elif "llama" in model_name_lower or "alpaca" in model_name_lower:
            model_type = ModelType.LLaMA
        elif "firefly" in model_name_lower:
            model_type = ModelType.Bloom
        else:
            model_type = ModelType.Unknown
        return model_type


class BaseLLMModel:
    def __init__(
        self,
        model_name,
        system_prompt="",
        temperature=1.0,
        top_p=1.0,
        n_choices=1,
        stop=None,
        presence_penalty=0,
        frequency_penalty=0,
        logit_bias=None,
        user="",
    ) -> None:

        self.history = []
        self.all_token_counts = []
        self.model_name = model_name
        self.model_type = ModelType.get_type(model_name)

        try:
            self.token_upper_limit = MODEL_TOKEN_LIMIT[model_name]
        except KeyError:
            self.token_upper_limit = DEFAULT_TOKEN_LIMIT

        self.interrupted = False
        self.system_prompt = system_prompt
        self.api_key = None
        self.need_api_key = False
        self.single_turn = False

        self.temperature = temperature
        self.top_p = top_p
        self.n_choices = n_choices
        self.stop_sequence = stop
        self.max_generation_token = None
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias
        self.user_identifier = user

    def get_answer_stream_iter(self):
        """stream predict, need to be implemented
        conversations are stored in self.history, with the most recent question, in OpenAI format
        should return a generator, each time give the next word (str) in the answer
        """
        logging.warning("stream predict not implemented, using at once predict instead")
        response, _ = self.get_answer_at_once()
        yield response

    def get_answer_at_once(self):
        """predict at once, need to be implemented
        conversations are stored in self.history, with the most recent question, in OpenAI format
        Should return:
        the answer (str)
        total token count (int)
        """
        logging.warning("at once predict not implemented, using stream predict instead")
        response_iter = self.get_answer_stream_iter()
        count = 0
        for response in response_iter:
            count += 1
        return response, sum(self.all_token_counts) + count

    def billing_info(self):
        """get billing infomation, inplement if needed"""
        logging.warning("billing info not implemented, using default")
        return BILLING_NOT_APPLICABLE_MSG

    def count_token(self, user_input):
        """get token count from input, implement if needed"""
        logging.warning("token count not implemented, using default")
        return len(user_input)

    def stream_next_chatbot(self, inputs, chatbot, fake_input=None, display_append=""):
        def get_return_value():
            return chatbot, status_text

        status_text = "开始实时传输回答……"
        if fake_input:
            chatbot.append((fake_input, ""))
        else:
            chatbot.append((inputs, ""))

        user_token_count = self.count_token(inputs)
        self.all_token_counts.append(user_token_count)
        logging.debug(f"输入token计数: {user_token_count}")

        stream_iter = self.get_answer_stream_iter()

        for partial_text in stream_iter:
            chatbot[-1] = (chatbot[-1][0], partial_text + display_append)
            self.all_token_counts[-1] += 1
            status_text = self.token_message()

            yield get_return_value()
            if self.interrupted:
                self.recover()
                break

        self.history.append(construct_assistant(partial_text))

    def next_chatbot_at_once(self, inputs, chatbot, fake_input=None, display_append=""):
        if fake_input:
            chatbot.append((fake_input, ""))
        else:
            chatbot.append((inputs, ""))

        if fake_input is not None:
            user_token_count = self.count_token(fake_input)
        else:
            user_token_count = self.count_token(inputs)

        self.all_token_counts.append(user_token_count)
        ai_reply, total_token_count = self.get_answer_at_once()
        self.history.append(construct_assistant(ai_reply))

        if fake_input is not None:
            self.history[-2] = construct_user(fake_input)
        chatbot[-1] = (chatbot[-1][0], ai_reply + display_append)

        if fake_input is not None:
            self.all_token_counts[-1] += count_token(construct_assistant(ai_reply))
        else:
            self.all_token_counts[-1] = total_token_count - sum(self.all_token_counts)

        status_text = self.token_message()
        return chatbot, status_text

    def prepare_inputs(self, real_inputs, use_websearch, reply_language, chatbot):
        display_append = []
        limited_context = False
        fake_inputs = real_inputs

        if use_websearch:
            limited_context = True
            search_results = ddg(real_inputs, max_results=5)

            reference_results = []
            for idx, result in enumerate(search_results):
                logging.debug(f"搜索结果{idx + 1}：{result}")
                domain_name = urllib3.util.parse_url(result["href"]).host
                reference_results.append([result["body"], result["href"]])
                display_append.append(
                    f"<li><a href=\"{result['href']}\" target=\"_blank\">{domain_name}</a></li>\n"
                )

            reference_results = add_source_numbers(reference_results)
            display_append = "<ol>\n\n" + "".join(display_append) + "</ol>"
            real_inputs = (
                replace_today(WEBSEARCH_PTOMPT_TEMPLATE)
                .replace("{query}", real_inputs)
                .replace("{web_results}", "\n\n".join(reference_results))
                .replace("{reply_language}", reply_language)
            )
        else:
            display_append = ""
        return limited_context, fake_inputs, display_append, real_inputs, chatbot

    def predict(
        self,
        inputs,
        chatbot,
        stream=False,
        use_websearch=False,
        reply_language="中文",
        should_check_token_count=True,
    ):  # repetition_penalty, top_k

        status_text = "开始生成回答……"
        logging.info(
            "输入为：" + colorama.Fore.BLUE + f"{inputs}" + colorama.Style.RESET_ALL
        )
        if should_check_token_count:
            yield chatbot + [(inputs, "")], status_text
        if reply_language == "跟随问题语言（不稳定）":
            reply_language = "the same language as the question, such as English, 中文, 日本語, Español, Français, or Deutsch."

        limited_context, fake_inputs, display_append, inputs, chatbot = self.prepare_inputs(
            real_inputs=inputs,
            use_websearch=use_websearch,
            reply_language=reply_language,
            chatbot=chatbot,
        )
        yield chatbot + [(fake_inputs, "")], status_text

        if (
            self.need_api_key and
            self.api_key is None
            and not shared.state.multi_api_key
        ):
            status_text = STANDARD_ERROR_MSG + NO_APIKEY_MSG
            logging.info(status_text)
            chatbot.append((inputs, ""))
            if len(self.history) == 0:
                self.history.append(construct_user(inputs))
                self.history.append("")
                self.all_token_counts.append(0)
            else:
                self.history[-2] = construct_user(inputs)
            yield chatbot + [(inputs, "")], status_text
            return
        elif len(inputs.strip()) == 0:
            status_text = STANDARD_ERROR_MSG + NO_INPUT_MSG
            logging.info(status_text)
            yield chatbot + [(inputs, "")], status_text
            return

        if self.single_turn:
            self.history = []
            self.all_token_counts = []
        self.history.append(construct_user(inputs))

        try:
            if stream:
                logging.debug("使用流式传输")
                iter = self.stream_next_chatbot(
                    inputs,
                    chatbot,
                    fake_input=fake_inputs,
                    display_append=display_append,
                )
                for chatbot, status_text in iter:
                    yield chatbot, status_text
            else:
                logging.debug("不使用流式传输")
                chatbot, status_text = self.next_chatbot_at_once(
                    inputs,
                    chatbot,
                    fake_input=fake_inputs,
                    display_append=display_append,
                )
                yield chatbot, status_text
        except Exception as e:
            status_text = STANDARD_ERROR_MSG + str(e)
            yield chatbot, status_text

        if len(self.history) > 1 and self.history[-1]["content"] != inputs:
            logging.info(
                "回答为："
                + colorama.Fore.BLUE
                + f"{self.history[-1]['content']}"
                + colorama.Style.RESET_ALL
            )

        if limited_context:
            # self.history = self.history[-4:]
            # self.all_token_counts = self.all_token_counts[-2:]
            self.history = []
            self.all_token_counts = []

        max_token = self.token_upper_limit - TOKEN_OFFSET

        if sum(self.all_token_counts) > max_token and should_check_token_count:
            count = 0
            while (
                sum(self.all_token_counts)
                > self.token_upper_limit * REDUCE_TOKEN_FACTOR
                and sum(self.all_token_counts) > 0
            ):
                count += 1
                del self.all_token_counts[0]
                del self.history[:2]

            logging.info(status_text)
            status_text = f"为了防止token超限，模型忘记了早期的 {count} 轮对话"
            yield chatbot, status_text

    def retry(
        self,
        chatbot,
        stream=False,
        use_websearch=False,
        reply_language="中文",
    ):
        logging.debug("重试中……")
        if len(self.history) > 0:
            inputs = self.history[-2]["content"]
            del self.history[-2:]
            self.all_token_counts.pop()
        elif len(chatbot) > 0:
            inputs = chatbot[-1][0]
        else:
            yield chatbot, f"{STANDARD_ERROR_MSG}上下文是空的"
            return

        iter = self.predict(
            inputs,
            chatbot,
            stream=stream,
            use_websearch=use_websearch,
            reply_language=reply_language,
        )
        for x in iter:
            yield x
        logging.debug("重试完毕")

    def interrupt(self):
        self.interrupted = True

    def recover(self):
        self.interrupted = False

    def set_token_upper_limit(self, new_upper_limit):
        self.token_upper_limit = new_upper_limit
        print(f"token上限设置为{new_upper_limit}")

    def set_temperature(self, new_temperature):
        self.temperature = new_temperature

    def set_top_p(self, new_top_p):
        self.top_p = new_top_p

    def set_n_choices(self, new_n_choices):
        self.n_choices = new_n_choices

    def set_stop_sequence(self, new_stop_sequence: str):
        new_stop_sequence = new_stop_sequence.split(",")
        self.stop_sequence = new_stop_sequence

    def set_max_tokens(self, new_max_tokens):
        self.max_generation_token = new_max_tokens

    def set_presence_penalty(self, new_presence_penalty):
        self.presence_penalty = new_presence_penalty

    def set_frequency_penalty(self, new_frequency_penalty):
        self.frequency_penalty = new_frequency_penalty

    def set_logit_bias(self, logit_bias):
        logit_bias = logit_bias.split()
        bias_map = {}
        encoding = tiktoken.get_encoding("cl100k_base")
        for line in logit_bias:
            word, bias_amount = line.split(":")
            if word:
                for token in encoding.encode(word):
                    bias_map[token] = float(bias_amount)
        self.logit_bias = bias_map

    def set_user_identifier(self, new_user_identifier):
        self.user_identifier = new_user_identifier

    def set_system_prompt(self, new_system_prompt):
        self.system_prompt = new_system_prompt

    def set_key(self, new_access_key):
        self.api_key = new_access_key.strip()
        msg = f"API密钥更改为了{hide_middle_chars(self.api_key)}"
        logging.info(msg)
        return msg

    def set_single_turn(self, new_single_turn):
        self.single_turn = new_single_turn

    def reset(self):
        self.history = []
        self.all_token_counts = []
        self.interrupted = False
        return [], self.token_message([0])

    def delete_first_conversation(self):
        if self.history:
            del self.history[:2]
            del self.all_token_counts[0]
        return self.token_message()

    def delete_last_conversation(self, chatbot):
        if len(chatbot) > 0 and STANDARD_ERROR_MSG in chatbot[-1][1]:
            chatbot.pop()
            return chatbot, self.history
        if len(self.history) > 0:
            self.history.pop()
            self.history.pop()
        if len(chatbot) > 0:
            chatbot.pop()
        if len(self.all_token_counts) > 0:
            self.all_token_counts.pop()
        msg = "删除了一组对话"
        return chatbot, msg

    def token_message(self, token_lst=None):
        if token_lst is None:
            token_lst = self.all_token_counts
        token_sum = 0
        for i in range(len(token_lst)):
            token_sum += sum(token_lst[: i + 1])
        return f"Token 计数: {sum(token_lst)}，本次对话累计消耗了 {token_sum} tokens"

    def save_chat_history(self, filename, chatbot, user_name):
        if filename == "":
            return
        if not filename.endswith(".json"):
            filename += ".json"
        return save_file(filename, self.system_prompt, self.history, chatbot, user_name)

    def export_markdown(self, filename, chatbot, user_name):
        if filename == "":
            return
        if not filename.endswith(".md"):
            filename += ".md"
        return save_file(filename, self.system_prompt, self.history, chatbot, user_name)

    def load_chat_history(self, filename, chatbot, user_name):
        logging.debug(f"{user_name} 加载对话历史中……")
        if type(filename) != str:
            filename = filename.name
        try:
            with open(os.path.join(HISTORY_DIR, user_name, filename), "r") as f:
                json_s = json.load(f)
            try:
                if type(json_s["history"][0]) == str:
                    logging.info("历史记录格式为旧版，正在转换……")
                    new_history = []
                    for index, item in enumerate(json_s["history"]):
                        if index % 2 == 0:
                            new_history.append(construct_user(item))
                        else:
                            new_history.append(construct_assistant(item))
                    json_s["history"] = new_history
                    logging.info(new_history)
            except:
                # 没有对话历史
                pass
            logging.debug(f"{user_name} 加载对话历史完毕")
            self.history = json_s["history"]
            return filename, json_s["system"], json_s["chatbot"]
        except FileNotFoundError:
            logging.warning(f"{user_name} 没有找到对话历史文件，不执行任何操作")
            return filename, self.system_prompt, chatbot
