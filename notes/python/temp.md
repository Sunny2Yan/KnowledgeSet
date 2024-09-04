## 1. model.train() 和 model.eval()

模型中有 BN 层 or Dropout层，需要在训练时添加 model.train()，在测试时添加 model.eval()。其中model.train() 是保证 BN 层用每一批数据的均值和方差，而 model.eval() 是保证 BN 用全部训练数据的均值和方差，若未加 model.eval() 则会造成实验结果震荡；而对于Dropout，model.train()是随机取一部分网络连接来训练更新参数，而 model.eval() 是利用到了所有网络连接。
