# Tokenizer

## 1. 方法
当前 tokenization 主要分为：word，sub-word， char-level 三个类型.

1. word: 分成单个字；
  缺点：
    1) vocabulary size太大；
    2) vocabulary 中存在较多相似的词；
    3) 面临严重的OOV（Out of vocabulary超出词汇表的词）问题。
2. sub-word：低频词拆分，高频词不拆分；如高频的boy不拆分，低频的boys拆分为boy和s；
   方法：WordPiece, Byte-Pair Encoding (BPE), Unigram, SentencePiece
3. char-level:

## 1.1 BPE--Byte Pair Encoding(llama, GPT)

1) 准备足够大的训练语料，并确定期望的Subword词表大小；
2) 将单词拆分为成最小单元。比如英文中26个字母加上各种符号，这些作为初始词表；
3) 在语料上统计单词内相邻单元对的频数，选取频数最高的单元对合并成新的Subword单元；
4) 重复第3步直到达到第1步设定的Subword词表大小或下一个最高频数为1.

```
for example:
    'low</w>': 5, 'lower</w>': 2, 'newest</w>': 6, 'widest</w>': 3
    1) l,o,w,e,r,n,s,t,i,d,</w>
    2) es出现6+3次，属于高频词需要合并 -> est -> est</w>; 此时词表为 l,o,w,r,n,i,d,est</w>
    3) 迭代至达到预设的词表大小，或最高词频出现频率为1
 ```

## 1.2 WordPiece(Bert)

WordPiece本质是同BPE，但在每次merge时，BPE选择频数最高的相邻子词合并，而WordPiece是选择能够最大化训练数据似然的合并。
设句子 $S=(t_1, t_2, \cdots, t_n)$ 由 $n$ 个子词组成，其中 $t_i$ 表示子词，且假设各个子词之间是独立存在的，则句子S的语言模型似然值等价于所有子词概率的乘积：

$$
\log P(S)=\sum_{i=1}^{n} \log P(t_i)
$$

假设把相邻位置的x和y两个子词进行合并，合并后产生的子词记为z，此时句子似然值的变化可表示为：

$$
\log P(t_z)-[\log P(t_x) + \log P(t_y)]=\log[\frac{P(t_z)}{P(t_x)P(t_y)}]
$$

## 1.3 ULM--Unigram Language Model()

先初始化一个大词表，根据评估准则不断丢弃词表，直到满足限定条件.
对于句子S，$x=(x_1, x_2, \cdots, x_n)$ 为句子的一个分词结果，由 $n$ 个子词组成。则当前分词下句子 $S$ 的似然值可以表示为：

$$
P(x)=\prod_{i=1}^{n} \log P(x_i)
$$

对于句子 $S$，选择似然值最大的作为分词结果，则可以表示为：

$$
x^*=\arg\max_{x\in U(x)} P(x)
$$

...

