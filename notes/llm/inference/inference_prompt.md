# LLM Inference Prompt

## 1. Input-Output

## 2. Chain of Thought
大模型逐步推理得到最终答案

```text
# one shot
这组数中的奇数加起来是偶数：4、8、9、15、12、2、1。
A：将所有奇数相加（9、15、1）得到25。答案为False。
这组数中的奇数加起来是偶数：15、32、5、13、82、7、1。
A：

# zero shot
我去市场买了10个苹果。我给了邻居2个苹果和修理工2个苹果。然后我去买了5个苹果并吃了1个。我还剩下多少苹果？
Let's think step by step  # 让我们逐步思考。

# 指定格式
Write a coherent passage of 4 short paragraphs. 

Make a plan then write. Your output should be of the following format:

Plan:
Your plan here.

Passage:
Your passage here.
```

## 3. Self Consistency
通过多次 CoT 采样多个不同的推理路径，采用投票的方式选取票数最多的结果

```text
# CoT prompt (eg: few shot)
Q：Olivia有23美元。她用每个3美元的价格买了五个百吉饼。她还剩多少钱？
A：她用每个3美元的价格买了5个百吉饼。这意味着她花了15美元。她还剩8美元。
Q：当我6岁时，我的妹妹是我的一半年龄。现在我70岁了，我的妹妹多大？
A：

# vote prompt
Given an instruction and several choices, decide which choice is most promising. Analyze each choice in detail, then conclude in the last line "The best choice is {s}", where s the integer id of the choice.
```

使用大模型推理多次，得出最终结果：
```text
A1：当我6岁时，我的妹妹是我的一半年龄，也就是3岁。现在我70岁了，所以她是70-3 = 67岁。答案是67。
A2：当叙述者6岁时，他的妹妹是他年龄的一半，也就是3岁。现在叙述者70岁了，他的妹妹应该是70-3 = 67岁。答案是67。
A3：当我6岁时，我的妹妹是我的一半年龄，也就是3岁。现在我70岁了，所以她是70/2 = 35岁。答案是35。

==> 67
```

# Tree of Thoughts
ToT 允许 LM 通过考虑多个不同的推理路径和自我评估选择来决定下一个动作过程来执行深思熟虑的决策，以及在做出全局选择时展望未来或回溯

![](/imgs/llm/inference/tot.png)

方法：
1. 问题分解（Thought decomposition）：将问题分解成多个中间步骤。每个步骤可以是短语、算式或计划。
2. 思维生成（Thought generator）：假设解决问题需要k个步骤，有下面两种方法生成推理内容。
   1) 采样（sample）：模型独立地从CoT提示中完整抽取k个推理内容
   2) 顺序生成（Propose）：顺序地使用prompt来逐步引导推理内容生成，每个推理内容都可能依赖于前一个推理内容
3. 评估（State evaluator）：评估哪些 thought 可行，哪些不可行。
   1) 投票（vote）
   2) 打分（value）
4. 搜索算法（Search algorithm）：BFS，DFS

将 ToT 框架的主要概念概括成了一段简短的提示词，指导 LLM 在一次提示中对中间思维做出评估：
```text
假设三位不同的专家来回答这个问题。
所有专家都写下他们思考这个问题的第一个步骤，然后与大家分享。
然后，所有专家都写下他们思考的下一个步骤并分享。
以此类推，直到所有专家写完他们思考的所有步骤。
只要大家发现有专家的步骤出错了，就让这位专家离开。
请问...
```