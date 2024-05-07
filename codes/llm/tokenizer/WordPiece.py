# -*- coding: utf-8 -*-
from collections import defaultdict
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


class WordPiece:
    special = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

    def __init__(self, corpus, vocab_size):
        self.corpus = corpus
        self.vocab_size = vocab_size

        word_freqs, self.vocab, splits = self.get_vocab()
        print("word_freqs:", word_freqs)
        print("vocab", self.vocab)
        print("splits:", splits)

        self.merge_pair(word_freqs, splits)
        print("vocab", self.vocab)

    def get_vocab(self):
        # step 1: 计算语料库中每个单词的频率
        word_freqs = defaultdict(int)
        for text in corpus:
            words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
                text)
            new_words = [word for word, offset in words_with_offsets]
            for word in new_words:
                word_freqs[word] += 1

        # 为单词的非首字母添加前缀
        alphabet = []
        for word in word_freqs.keys():
            if word[0] not in alphabet:
                alphabet.append(word[0])
            for letter in word[1:]:
                if f"##{letter}" not in alphabet:
                    alphabet.append(f"##{letter}")

        # step 3: 添加特殊token
        vocab = self.special + alphabet.copy()

        # step 4: 对word分割
        splits = {
            word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
            for word in word_freqs.keys()
        }

        return word_freqs, vocab, splits

    @staticmethod
    def compute_pair_scores(word_freqs, splits):
        """计算每个pair的分数"""
        letter_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                letter_freqs[split[0]] += freq
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                letter_freqs[split[i]] += freq
                pair_freqs[pair] += freq
            letter_freqs[split[-1]] += freq

        scores = {
            pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
            for pair, freq in pair_freqs.items()
        }

        return scores

    @staticmethod
    def _merge_pair(a, b, word_freqs, splits):
        for word in word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    merge = a + b[2:] if b.startswith("##") else a + b
                    split = split[:i] + [merge] + split[i + 2:]
                else:
                    i += 1
            splits[word] = split
        return splits

    def merge_pair(self, word_freqs, splits):
        while len(self.vocab) < self.vocab_size:
            scores = self.compute_pair_scores(word_freqs, splits)
            best_pair, max_score = "", None
            for pair, score in scores.items():
                if max_score is None or max_score < score:
                    best_pair = pair
                    max_score = score
            print(best_pair, max_score)

            splits = self._merge_pair(*best_pair, word_freqs, splits)
            new_token = (
                best_pair[0] + best_pair[1][2:]
                if best_pair[1].startswith("##")
                else best_pair[0] + best_pair[1]
            )
            self.vocab.append(new_token)

    def encode_word(self, word):
        """从第一个词的开头寻找最大的子词并将其拆分"""
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in self.vocab:
                i -= 1
            if i == 0:
                return ["[UNK]"]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f"##{word}"
        return tokens

    def tokenize(self, text):
        pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(
            text)
        pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        encoded_words = [self.encode_word(word) for word in pre_tokenized_text]
        return sum(encoded_words, [])


if __name__ == '__main__':
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]
    word_piece = WordPiece(corpus, 70)
    print(word_piece)