# -*- coding: utf-8 -*-

import re
import time
import json
import warnings
import random
import torch
import evaluate
import transformers


class RougeEval:
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    @staticmethod
    def load_dataset(data_path: str):
        with open(data_path, 'r') as files:
            files = json.loads(files.read())
            for file in files:
                yield file

    @staticmethod
    def load_model(model_path: str):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map='auto')
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

        return model, tokenizer

    def evaluate_model(self, data_path: str, model_path: str,
                       metric_type='rouge', cache_dir=None, threshold=10):
        dataset = self.load_dataset(data_path)
        model, tokenizer = self.load_model(model_path)
        metric = evaluate.load(metric_type, cache_dir=cache_dir)

        result = {"num_tokens": [], "out_tokens": [], "metric": []}

        for n, data in enumerate(dataset):
            if n == threshold: break

            print(data['_id'])
            query_text = data['report'] + f"{self.B_INST} {'Please make a summary of the above report.'} {self.E_INST}"

            inputs = tokenizer(query_text, return_tensors="pt")
            num_tokens = len(inputs.input_ids[0])
            result['num_tokens'].append(num_tokens)
            generate_ids = model.generate(inputs.input_ids,
                                          max_new_tokens=512,
                                          temperature=0.6,
                                          top_p=0.9)
            response = tokenizer.batch_decode(generate_ids)[0][
                       len(query_text)+3:]
            # print(response)

            metric_score = metric.compute(predictions=[response],
                                          references=[data['summary']])
            print(metric_score)

            result['out_tokens'].append(
                len(generate_ids[0]) - len(inputs.input_ids[0]))
            result['metric'].append(metric_score)

        return result


if __name__ == '__main__':
    data_path = "/data/persist/datasets/gov_report/split/1k.json"
    cache_dir = '/data/persist/datasets/temp/'
    model_path = "/data/persist/models/llama2-7b-chat"

    rouge_eval = RougeEval()
    result = rouge_eval.evaluate_model(
        data_path, model_path, cache_dir=cache_dir)
    print(result)