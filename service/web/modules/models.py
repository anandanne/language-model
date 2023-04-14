import colorama

from . import config
from .base_model import BaseLLMModel, ModelType
from .utils import *


class OpenAIClient(BaseLLMModel):
    def __init__(
        self,
        model_name,
        api_key,
        system_prompt=INITIAL_SYSTEM_PROMPT,
        temperature=1.0,
        top_p=1.0,
    ) -> None:
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            system_prompt=system_prompt,
        )
        self.api_key = api_key
        self.need_api_key = True
        self._refresh_header()

    def get_answer_stream_iter(self):
        response = self._get_response(stream=True)
        if response is not None:
            iter = self._decode_chat_response(response)
            partial_text = ""
            for i in iter:
                partial_text += i
                yield partial_text
        else:
            yield STANDARD_ERROR_MSG + GENERAL_ERROR_MSG

    def get_answer_at_once(self):
        response = self._get_response()
        response = json.loads(response.text)
        content = response["choices"][0]["message"]["content"]
        total_token_count = response["usage"]["total_tokens"]
        return content, total_token_count

    def count_token(self, user_input):
        input_token_count = count_token(construct_user(user_input))
        if self.system_prompt is not None and len(self.all_token_counts) == 0:
            system_prompt_token_count = count_token(
                construct_system(self.system_prompt)
            )
            return input_token_count + system_prompt_token_count
        return input_token_count

    def billing_info(self):
        try:
            curr_time = datetime.datetime.now()
            last_day_of_month = get_last_day_of_month(curr_time).strftime("%Y-%m-%d")
            first_day_of_month = curr_time.replace(day=1).strftime("%Y-%m-%d")
            usage_url = f"{shared.state.usage_api_url}?start_date={first_day_of_month}&end_date={last_day_of_month}"
            try:
                usage_data = self._get_billing_data(usage_url)
            except Exception as e:
                logging.error(f"获取API使用情况失败:" + str(e))
                return f"**获取API使用情况失败**"
            rounded_usage = "{:.5f}".format(usage_data["total_usage"] / 100)
            return f"**本月使用金额** \u3000 ${rounded_usage}"
        except requests.exceptions.ConnectTimeout:
            status_text = (
                    STANDARD_ERROR_MSG + CONNECTION_TIMEOUT_MSG + ERROR_RETRIEVE_MSG
            )
            return status_text
        except requests.exceptions.ReadTimeout:
            status_text = STANDARD_ERROR_MSG + READ_TIMEOUT_MSG + ERROR_RETRIEVE_MSG
            return status_text
        except Exception as e:
            logging.error(f"获取API使用情况失败:" + str(e))
            return STANDARD_ERROR_MSG + ERROR_RETRIEVE_MSG

    def set_token_upper_limit(self, new_upper_limit):
        pass

    def set_key(self, new_access_key):
        self.api_key = new_access_key.strip()
        self._refresh_header()
        msg = f"API密钥更改为了{hide_middle_chars(self.api_key)}"
        logging.info(msg)
        return msg

    @shared.state.switching_api_key  # 在不开启多账号模式的时候，这个装饰器不会起作用
    def _get_response(self, stream=False):
        openai_api_key = self.api_key
        system_prompt = self.system_prompt
        history = self.history
        logging.debug(colorama.Fore.YELLOW + f"{history}" + colorama.Fore.RESET)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}",
        }

        if system_prompt is not None:
            history = [construct_system(system_prompt), *history]

        payload = {
            "model": self.model_name,
            "messages": history,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n_choices,
            "stream": stream,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }

        if self.max_generation_token is not None:
            payload["max_tokens"] = self.max_generation_token
        if self.stop_sequence is not None:
            payload["stop"] = self.stop_sequence
        if self.logit_bias is not None:
            payload["logit_bias"] = self.logit_bias
        if self.user_identifier is not None:
            payload["user"] = self.user_identifier

        if stream:
            timeout = TIMEOUT_STREAMING
        else:
            timeout = TIMEOUT_ALL

        # 如果有自定义的api-host，使用自定义host发送请求，否则使用默认设置发送请求
        if shared.state.completion_url != COMPLETION_URL:
            logging.info(f"使用自定义API URL: {shared.state.completion_url}")

        with retrieve_proxy():
            try:
                response = requests.post(
                    shared.state.completion_url,
                    headers=headers,
                    json=payload,
                    stream=stream,
                    timeout=timeout,
                )
            except:
                return None
        return response

    def _refresh_header(self):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _get_billing_data(self, billing_url):
        with retrieve_proxy():
            response = requests.get(
                billing_url,
                headers=self.headers,
                timeout=TIMEOUT_ALL,
            )

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            raise Exception(
                f"API request failed with status code {response.status_code}: {response.text}"
            )

    def _decode_chat_response(self, response):
        error_msg = ""
        for chunk in response.iter_lines():
            if chunk:
                chunk = chunk.decode()
                chunk_length = len(chunk)
                try:
                    chunk = json.loads(chunk[6:])
                except json.JSONDecodeError:
                    print(f"JSON解析错误,收到的内容: {chunk}")
                    error_msg += chunk
                    continue
                if chunk_length > 6 and "delta" in chunk["choices"][0]:
                    if chunk["choices"][0]["finish_reason"] == "stop":
                        break
                    try:
                        yield chunk["choices"][0]["delta"]["content"]
                    except Exception as e:
                        # logging.error(f"Error: {e}")
                        continue
        if error_msg:
            raise Exception(error_msg)


class FastAPIClient(BaseLLMModel):

    api_url = None

    def get_answer_at_once(self):
        response = self._get_response()
        response = json.loads(response.text)
        response = response["choices"][0]["message"]["content"]
        total_token_count = response["usage"]["total_tokens"]

        return response, total_token_count

    def get_answer_stream_iter(self):
        response = self._get_response(stream=True)
        if response is not None:
            iter = self._decode_chat_response(response)
            for i in iter:
                yield i
        else:
            yield STANDARD_ERROR_MSG + GENERAL_ERROR_MSG

    @shared.state.switching_api_key  # 在不开启多账号模式的时候，这个装饰器不会起作用
    def _get_response(self, stream=False):
        history = self.history
        logging.debug(colorama.Fore.YELLOW + f"{history}" + colorama.Fore.RESET)
        headers = {
            "Content-Type": "application/json",
        }

        payload = {
            "messages": history,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n_choices,
            "stream": stream,
        }

        if self.max_generation_token is not None:
            payload["max_tokens"] = self.max_generation_token
        if self.stop_sequence is not None:
            payload["stop"] = self.stop_sequence

        if stream:
            timeout = TIMEOUT_STREAMING
        else:
            timeout = TIMEOUT_ALL

        with retrieve_proxy():
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    stream=stream,
                    timeout=timeout,
                )
            except:
                return None
        return response

    def _refresh_header(self):
        self.headers = {
            "Content-Type": "application/json",
        }

    def _decode_chat_response(self, response):
        for chunk in response.iter_lines():
            if chunk:
                chunk = chunk.decode()
                chunk_length = len(chunk)
                try:
                    chunk = chunk[6:]
                    if chunk == "[DONE]":
                        continue
                    else:
                        chunk = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                if chunk_length > 6 and "message" in chunk["choices"][0]:
                    if chunk["choices"][0]["finish_reason"] == "stop":
                        break
                    try:
                        yield chunk["choices"][0]["message"]["content"]
                    except Exception as e:
                        # logging.error(f"Error: {e}")
                        continue


class ChatGLMClient(FastAPIClient):

    api_url = CHATGLM_COMPLETION_URL


class LLaMAClient(FastAPIClient):

    api_url = LLAMA_COMPLETION_URL
    lora_path = None


class FireflyClient(FastAPIClient):

    api_url = FIREFLY_COMPLETION_URL


def get_model(
    model_name,
    lora_model_path=None,
    access_key=None,
    temperature=None,
    top_p=None,
    system_prompt=None,
) -> BaseLLMModel:
    msg = f"模型设置为了： {model_name}"
    model_type = ModelType.get_type(model_name)
    lora_selector_visibility = False
    lora_choices = []
    dont_change_lora_selector = False
    if model_type != ModelType.OpenAI:
        config.local_embedding = True
    # del current_model.model
    model = None
    try:
        if model_type == ModelType.OpenAI:
            logging.info(f"正在加载OpenAI模型: {model_name}")
            model = OpenAIClient(
                model_name=model_name,
                api_key=access_key,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
            )
        elif model_type == ModelType.ChatGLM:
            logging.info(f"正在加载ChatGLM模型: {model_name}")
            model = ChatGLMClient(model_name)
        elif model_type == ModelType.Bloom:
            logging.info(f"正在加载ChatGLM模型: {model_name}")
            model = FireflyClient(model_name)
        elif model_type == ModelType.LLaMA and lora_model_path == "":
            msg = f"现在请为 {model_name} 选择LoRA模型"
            logging.info(msg)
            lora_selector_visibility = True
            if os.path.isdir("lora"):
                lora_choices = get_file_names("lora", plain=True, filetypes=[""])
            lora_choices = ["No LoRA"] + lora_choices
        elif model_type == ModelType.LLaMA and lora_model_path != "":
            logging.info(f"正在加载LLaMA模型: {model_name} + {lora_model_path}")
            dont_change_lora_selector = True
            if lora_model_path == "No LoRA":
                lora_model_path = None
                msg += " + No LoRA"
            else:
                msg += f" + {lora_model_path}"
            model = LLaMAClient(model_name, lora_model_path)
        elif model_type == ModelType.Unknown:
            raise ValueError(f"未知模型: {model_name}")
        logging.info(msg)
    except Exception as e:
        logging.error(e)
        msg = f"{STANDARD_ERROR_MSG}: {e}"
    if dont_change_lora_selector:
        return model, msg
    else:
        return model, msg, gr.Dropdown.update(choices=lora_choices, visible=lora_selector_visibility)
