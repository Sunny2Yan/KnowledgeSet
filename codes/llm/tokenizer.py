# -*- coding: utf-8 -*-
import re
import toolz
from collections import Counter


# 1.处理数据
def wordpunct_tokenize(text):
    """分词器，将句子拆分成单词（如根据空格、标点进行拆分)"""
    _pattern = r"\w+|[^\w\s]+"
    _regexp = re.compile(_pattern, flags=re.UNICODE | re.MULTILINE | re.DOTALL)
    return _regexp.findall(text)


class BPETokenizer():
    special = ['<UNK>', '<PAD>', '<END>', '<MASK>']

    def __init__(self, vocab_size=1000, lowercase=True,
                 basic_tokenizer=wordpunct_tokenize):
        self.lowercase = lowercase
        self.vocab_size = vocab_size
        self.basic_tokenizer = basic_tokenizer

    # 2.词典训练
    def fit(self, corpus: list, max_steps=10000, out_fn='vocab.txt'):
        """分词器训练，返回训练得到的vocabulary."""

        # 1) 统计初始词典
        if self.lowercase:
            corpus = [s.lower() for s in corpus]
        word_corpus = Counter([tuple(data) + ("</w>",) for data in
                               toolz.concat(map(self.basic_tokenizer, corpus))])
        vocab = self._count_vocab(word_corpus)
        print(vocab)

        # 2) 逐步合并初始词典中的高频二元组
        for i in range(max_steps):
            word_corpus, bi_cnt = self._fit_step(word_corpus)
            vocab = self._count_vocab(word_corpus)
            if len(vocab) >= self.vocab_size or bi_cnt < 0: break

        # 3) 将一些特殊词加入最终的词典
        for s in self.special:
            if s not in vocab:
                vocab.insert(0, (s, 99999))

        # 4) 导出词典
        # with open(out_fn, 'w') as f:
        #     f.write('\n'.join([w for w, _ in vocab]))
        self.vocab = [token for token, _ in vocab]

        return vocab

    def _count_vocab(self, word_corpus):
        # 对于列表嵌套列表，拆开子列表，不做去重和排序，返回迭代器
        # eg: list(concat([[], [1], [2, 3]])) -> [1, 2, 3]
        _r = Counter([data for data in toolz.concat(
            [word * cnt for word, cnt in word_corpus.items()])])
        _r = sorted(_r.items(), key=lambda x: -x[1])
        return _r

    def _fit_step(self, word_corpus):
        ngram = 2
        bigram_counter = Counter()

        # step1: 以步长1，窗口尺寸2，在每个单词上滚动，统计二元组频次
        for token, count in word_corpus.items():
            if len(token) < 2: continue
            # 用窗口遍历
            for bigram in toolz.sliding_window(ngram, token):
                bigram_counter[bigram] += count

        # step2: 选出频次最大的二元组
        if len(bigram_counter) > 0:
            max_bigram = max(bigram_counter, key=bigram_counter.get)
        else:
            return word_corpus, -1
        bi_cnt = bigram_counter.get(max_bigram)

        # step3: 从corpus中将最大二元组出现的地方替换成一个token.
        for token in list(word_corpus):
            _new_token = tuple(' '.join(token).replace(
                ' '.join(max_bigram), ''.join(max_bigram)).split(' '))
            if _new_token != token:
                word_corpus[_new_token] = word_corpus[token]
                word_corpus.pop(token)
        return word_corpus, bi_cnt

    # 3.分词
    def tokenize(self, text: str, add_post='</w>'):
        """将text转换成tokens."""
        all_tokens = []
        if self.lowercase:
            text = text.lower()
        new_token = []

        # step1: 简单分词，并遍历token
        for token in self.basic_tokenizer(text):
            token = list(token)
            if add_post:
                token = token + [add_post]
            start, end = 0, len(token)

            # 查找最长sub_token
            while start < end:
                sub_token = ''.join(token[start:end])
                if sub_token in self.vocab:
                    new_token.append(sub_token)
                    start = end
                    end = len(token)
                elif end - start == 1:
                    new_token.append('<UNK>')
                    start = end
                    end = len(token)
                else:
                    end -= 1
        all_tokens.extend(new_token)
        return all_tokens

    def encode(self, text: str):
        """将text转换成token_ids."""
        tokens_list = self.tokenize(text)
        ids_list = list(map(self._token2id, tokens_list))
        # ids_list = [list(map(lambda x: self._token2id(x), tokens)) for tokens in
        #             tokens_list]
        return ids_list

    def decode(self, token_ids):
        """将token_ids还原成text."""
        sentences = []
        for ids in token_ids:
            sentence = list(map(lambda x: self._id2token(x), ids))
            sentence = ''.join(sentence).replace('</w>', ' ')
            sentences.append(sentence)
        return sentences

    def _token2id(self, token):
        # print('vocab:', self.vocab)
        if token in self.vocab:
            return self.vocab.index(token)
        return self.vocab.index('<UNK>')

    def _id2token(self, id):
        return self.vocab[id]

bpe = BPETokenizer()
text = """
Contrastive learning aims to learn good representations by comparing positive 
and negative examples. CLIP [Radford et al., 2021] and SimCLR [Chen et al., 
2020] are two popular contrastive learning methods that have achieved 
state-of-the-art performance in the image domain. 
"""

text = text.replace('\n', '').split(' ')

vocab = bpe.fit(text)
print(vocab)

x = "LMs with memory augmentation, which uses in-batch negatives to improve the quality of representations"
print(bpe.encode(x))

print(bpe.decode([123, 21, 324, 343, 56, 22, 49]))


# 调包实现
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

tokenizer.pre_tokenizer = Whitespace()  # 清空预置的词
# files = [...]
tokenizer.train(['./datasets/timemechine.txt'], trainer)
tokenizer.save('tokenizer.json')