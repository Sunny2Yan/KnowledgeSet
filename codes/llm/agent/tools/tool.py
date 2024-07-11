# -*- coding: utf-8 -*-
from abc import abstractmethod


class Tool:
    def __init__(self, description, name, parameters):
        self.type = "function"
        self.description = description
        self.name = name
        self.parameters = parameters

    @abstractmethod
    def func(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @abstractmethod
    def load_from_dict(cls, dict_data):
        pass