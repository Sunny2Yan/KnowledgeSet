# -*- coding: utf-8 -*-
import os
import json
import torch
from copy import deepcopy
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from core.profile import SINGLE_MESSAGE_TEMPLATE, SUMMARY_PROMPT_TEMPLATE


class MemoryStorage:
    """存储在内存中"""

    def __init__(self) -> None:
        self.memory_list: List[Dict] = []

    def __len__(self) -> int:
        return len(self.memory_list)

    def save(self, records: List[Dict[str, Any]]) -> None:
        self.memory_list.extend(deepcopy(records))

    def load(self) -> List[Dict[str, Any]]:
        return deepcopy(self.memory_list)

    def clear(self) -> None:
        self.memory_list.clear()


class JsonStorage:
    """存储到json文件中"""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.json_path = path or Path("./chat_history.json")
        self.json_path.touch()

    def save(self, records: List[Dict[str, Any]]) -> None:
        with self.json_path.open("a", encoding='utf-8') as f:
            f.writelines([json.dumps(r, cls=json.JSONEncoder) + "\n" for r in records])

    def load(self) -> List[Dict[str, Any]]:
        with self.json_path.open("r") as f:
            return [json.loads(r) for r in f.readlines()]

    def clear(self) -> None:
        with self.json_path.open("w"):
            pass


class Memory(ABC):

    @abstractmethod
    def get_memory(self):
        pass

    @staticmethod
    def encode_memory(massages: list, agent_name=None):
        """将messages转换成一个字符串"""
        encoded_memory = ""
        for message in massages:
            name = message.get("name")
            if agent_name and name and agent_name == name:
                name = f"you({agent_name})"
            role, content = message["role"], message["content"]
            single_message = SINGLE_MESSAGE_TEMPLATE.format(
                name=name, role=role, content=content
            )
            encoded_memory += "\n" + single_message
        return encoded_memory

    @staticmethod
    def get_relevant_memory(query: str, history: list,
                            embeddings: torch.Tensor, top_k: int = 5):
        """使用语义搜索获取关键的历史信息。"""

        relevant_memory = []
        query_embedding = get_embedding(query)
        hits = semantic_search(
            query_embedding, embeddings, top_k=min(top_k, embeddings.shape[0])
        )
        hits = hits[0]
        for hit in hits:
            matching_idx = hit["corpus_id"]
            try:
                relevant_memory.append(history[matching_idx])
            except:
                return []
        return relevant_memory


class ShortTermMemory(Memory):

    def __init__(
        self,
        config: Optional[dict] = None,
        messages: Optional[list] = None,
        window_size: Optional[int] = None,
    ):
        self.config = config if config else {}
        self.storage = MemoryStorage()
        # messages = [{"name": "" , "role": "", "content": ""}]
        self.storage.save(messages if messages else [])
        self.window_size = window_size  # 指定最近聊天的次数

    def __len__(self):
        return len(self.storage)

    def get_memory(self):
        return self.storage.load()

    def append_memory(self, message):
        self.storage.save([message])

    def update_memory(self, config, messages):
        self.config = config
        self.storage.clear()
        self.storage.save(messages)

    def get_memory_string(self, agent_name=None):
        encoded_memory = Memory.encode_memory(self.get_memory(), agent_name)
        return encoded_memory

    def get_memory_embedding(self):
        encoded_memory = self.get_memory_string()
        embed = get_embedding(encoded_memory)
        return embed

    def get_memory_summary(self):
        encoded_memory = self.get_memory_string()

        # Using GPT-3.5 to summarize the conversation
        response = completion_with_backoff(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": SUMMARY_PROMPT_TEMPLATE.format(
                        conversation=encoded_memory
                    ),
                },
            ],
            temperature=0,
        )
        summary = response.choices[0].message["content"].strip()
        return summary

    def to_dict(self):
        return {"config": self.config, "memory": self.get_memory(), }

    @staticmethod
    def load_from_json(json_dict):
        return ShortTermMemory(
            config=json_dict["config"],
            messages=json_dict["memory"], )


class LongTermMemory(Memory):

    def __init__(
        self,
        config: dict,
        json_path: str,
        chunk_list: Optional[list] = None,
        storage: JsonStorage = None,
        window_size: Optional[int] = 3,
    ):
        self.config = config
        self.json_path = json_path
        if storage:
            self.storage = storage
        else:
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            self.storage = JsonStorage(Path(json_path))
            self.storage.save(chunk_list if chunk_list else [])
        self.window_size = window_size

    def get_memory(self):
        return self.storage.load()

    def update_memory(self, config, chunk_list):
        self.config = config
        self.storage.clear()
        self.storage.save(chunk_list)

    def append_memory(self, chunk):
        self.storage.save([chunk])

    def append_memory_from_short_term_memory(self, short_term_memory: ShortTermMemory):
        if len(short_term_memory) >= self.window_size:
            memory = short_term_memory.get_memory()
            self.storage.save(
                [Memory.encode_memory(memory[-self.window_size :])])

    def to_dict(self):
        return {
            "config": self.config,
            "memory": self.get_memory(), }

    @staticmethod
    def load_from_json(json_dict):
        return LongTermMemory(
            config=json_dict["config"],
            chunk_list=json_dict["memory"],
        )