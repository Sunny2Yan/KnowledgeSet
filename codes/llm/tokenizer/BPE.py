# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import re
from collections import defaultdict, Counter


class BPE:
    def __init__(self):
        ...

    def extract_frequencies(self, sequence):
        """给定一个字符串，计算字符串中的单词出现的频率，并返回词表（一个词到频率的映射字典）。
        """
        token_counter = Counter()
        for item in sequence:
            tokens = ' '.join(list(item)) + ' </w>'
            token_counter[tokens] += 1

        return token_counter

    def frequency_of_pairs(self, frequencies):
        """给定一个词频字典，返回一个从字符对到频率的映射字典。
        """
        pairs_count = Counter()
        for token, count in frequencies.items():
            chars = token.split()
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i+1])
                pairs_count[pair] += count

        return pairs_count

    def merge_vocab(self, merge_pair, vocab):
        """给定一对相邻词元和一个词频字典，将相邻词元合并为新的词元，并返回新的词表。
        """
        re_pattern = re.escape(' '.join(merge_pair))
        pattern = re.compile(r'(?<!\S)' + re_pattern + r'(?!\S)')
        updated_tokens = {pattern.sub(''.join(merge_pair), token): freq for token, freq in vocab.items()}

        return updated_tokens

    def encode_with_bpe(self, texts, iterations):
        """给定待分词的数据以及最大合并次数，返回合并后的词表。
        """
        vocab_map = self.extract_frequencies(texts)
        for _ in range(iterations):
            pair_freqs = self.frequency_of_pairs(vocab_map)
            if not pair_freqs:
                break
            most_common_pair = pair_freqs.most_common(1)[0][0]
            vocab_map = self.merge_vocab(most_common_pair, vocab_map)

        return vocab_map


if __name__ == '__main__':
    text = "This section presents experiments to investigate the effect of different modeling choices."
    vocab_size = 1000

    bpe = BPE()
    print(bpe.encode_with_bpe(text, vocab_size))

