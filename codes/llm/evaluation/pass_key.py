# -*- coding: utf-8 -*-

import re
import random
import torch
import transformers


class PassKeyEval:
    @staticmethod
    def load_model(model_path):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='auto')

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

        return model, tokenizer

    @staticmethod
    def generate_prompt(n_garbage):
        """生成一个文本文件，并在随机位置插入一个执行行.
        """
        n_garbage_prefix = random.randint(0, n_garbage)
        n_garbage_suffix = n_garbage - n_garbage_prefix

        task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
        garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
        garbage_inf = " ".join([garbage] * 2000)

        assert len(garbage_inf) >= n_garbage
        garbage_prefix = garbage_inf[:n_garbage_prefix]
        garbage_suffix = garbage_inf[:n_garbage_suffix]
        pass_key = random.randint(1, 50000)
        information_line = f"The pass key is {pass_key}. " \
                           f"Remember it. {pass_key} is the pass key."
        final_question = "What is the pass key?"
        lines = [task_description, garbage_prefix, information_line,
                 garbage_suffix, final_question]

        return "\n".join(lines), pass_key  # prompt, number

    def evaluate_model(self, model_path, num_tests=10):
        result = {"num_tokens": 0, "accuracy": 0}

        n_values = [0, 100, 500, 1000, 5000, 8000, 10000, 12000, 14000, 18000,
                    20000, 25000, 38000]

        for n in n_values:
            print(f"the garbage length is {n}.")
            model_metric = {"num_tokens": 0, "accuracy": 0}

            for i in range(num_tests):
                prompt_text, pass_key = self.generate_prompt(n)
                query_text = f"[INST] {prompt_text.strip()} [/INST] Pass key is "

                model, tokenizer = self.load_model(model_path)
                num_tokens = len(tokenizer.encode(prompt_text))
                inputs = tokenizer(query_text, return_tensors="pt")

                model_metric['num_tokens'] += num_tokens
                generate_ids = model.generate(inputs.input_ids,
                                              max_new_tokens=30,
                                              temperature=0.6,
                                              top_p=0.9)

                response = tokenizer.batch_decode(generate_ids)[0][
                           len(query_text)+4:]
                try:
                    predict_pass_key = int(re.search(r'\d+', response).group())
                    model_metric['accuracy'] += (pass_key == predict_pass_key)
                except Exception:
                    pass

                print((num_tokens, pass_key, response, model_metric['accuracy']))

            result['num_tokens'] = int(model_metric['num_tokens'] / 10)
            result["accuracy"] = model_metric['accuracy'] / 10
            print(result)

        return result


if __name__ == '__main__':
    model_path = "/data/persist/models/llama2-7b-chat"

    eval_model = PassKeyEval()
    result = eval_model.evaluate_model(model_path)
    print(result)