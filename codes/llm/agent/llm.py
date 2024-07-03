# -*- coding: utf-8 -*-
import os
import time
# import litellm
# import backoff
from abc import abstractmethod
from typing import Union

# from ..utils.files import save_logs
# from ..utils.config import Config

WAIT_TIME = 20


class LLMConfig(Config):

    required_fields = []

    def __init__(self, config_path_or_dict: Union[str, dict] = None):
        super().__init__(config_path_or_dict)
        self._validate_config()

        self.LLM_type: str = self.config_dict.get("LLM_type", "OpenAI")
        self.model: str = self.config_dict.get("model", "gpt-4-turbo-2024-04-09")
        self.temperature: float = self.config_dict.get("temperature", 0.3)
        self.log_path: str = self.config_dict.get("log_path", "logs")
        self.API_KEY: str = self.config_dict.get(
            "OPENAI_API_KEY", os.environ["OPENAI_API_KEY"]
        )
        self.API_BASE = self.config_dict.get(
            "OPENAI_BASE_URL", os.environ.get("OPENAI_BASE_URL")
        )
        self.MAX_CHAT_MESSAGES: int = self.config_dict.get("max_chat_messages", 10)
        self.ACTIVE_MODE: bool = self.config_dict.get("ACTIVE_MODE", False)
        self.SAVE_LOGS: bool = self.config_dict.get("SAVE_LOGS", False)


class LLM:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.model = self.config.model
        self.temperature = self.config.temperature
        self.log_path = self.config.log_path
        self.API_KEY = self.config.API_KEY
        self.API_BASE = self.config.API_BASE
        self.MAX_CHAT_MESSAGES = self.config.MAX_CHAT_MESSAGES
        self.ACTIVE_MODE = self.config.ACTIVE_MODE
        self.SAVE_LOGS = self.config.SAVE_LOGS

    @abstractmethod
    def get_response(cls, **kwargs):
        pass


class OpenAILLM(LLM):

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        assert self.config.LLM_type == "OpenAI"

    def get_stream(self, response, log_path, messages):
        ans = ""
        for res in response:
            if res:
                r = (
                    res.choices[0]["delta"].get("content")
                    if res.choices[0]["delta"].get("content")
                    else ""
                )
                ans += r
                yield r

        if self.SAVE_LOGS:
            save_logs(log_path, messages, ans)

    def get_response(
        self,
        chat_messages,
        system_prompt,
        last_prompt=None,
        stream=False,
        tools=None,
        tool_choice="auto",
        response_format=None,
        **kwargs,
    ):
        """
        return LLM's response
        """
        litellm.api_key = self.API_KEY
        if self.API_BASE:
            litellm.api_base = self.API_BASE

        messages = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        )

        if chat_messages:
            if len(chat_messages) > self.MAX_CHAT_MESSAGES:
                chat_messages = chat_messages[-self.MAX_CHAT_MESSAGES :]
            if isinstance(chat_messages[0], dict):
                messages += chat_messages

        if last_prompt:
            if self.ACTIVE_MODE:
                last_prompt += " Be accurate but concise in response."

            if messages:
                # messages[-1]["content"] += last_prompt
                messages.append({"role": "user", "content": last_prompt})
            else:
                messages = [{"role": "user", "content": last_prompt}]

        if tools:
            response = completion_with_backoff(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=self.temperature,
                response_format=response_format,
                custom_llm_provider="openai",
            )
        else:
            response = completion_with_backoff(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stream=stream,
                response_format=response_format,
                custom_llm_provider="openai",
            )

        if response.choices[0].message.get("tool_calls"):
            content = response.choices[0].message
        elif stream:
            content = self.get_stream(response, self.log_path, messages)
        else:
            content = response.choices[0].message["content"].strip()
            if self.SAVE_LOGS:
                save_logs(self.log_path, messages, content)

        return response, content