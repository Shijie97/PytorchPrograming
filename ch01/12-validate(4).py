# !user/bin/python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

# 看看torch中的torch.nn.embedding
# embedding接收两个参数
# 第一个是num_embeddings，它表示词库的大小，则所有词的下标从0 ~ num_embeddings-1
# 第二个是embedding_dim，表示词嵌入维度
# 词嵌入层有了上面这两个必须有的参数，就形成了类，这个类可以有输入和输出
# 输入的数据结构不限，但是数据结构里面每个单元的元素必须指的是下标，即要对应0 ~ num_embeddings-1
# 输出的数据结构和输入一样，只不过将下标换成对应的词嵌入
# 最开始的时候词嵌入的矩阵是随机初始化的，但是作为嵌入层，会不断的学习参数，所以最后训练完成的参数一定是学习完成的
# embedding层还可以接受一个可选参数padding_idx，这个参数指定的维度，但凡输入的时候有这个维度，输出一律填0

# 下面来看一下吧
embedding = nn.Embedding(10, 3)
inputs = torch.tensor([[1, 2, 4, 5],
                       [4, 3, 2, 9]])
print(embedding(inputs))

# tensor([[[ 0.3721,  0.3502,  0.8029],
#          [-0.2410,  0.0723, -0.6451],
#          [-0.4488,  1.4382,  0.1060],
#          [-0.1430, -0.8969,  0.7086]],
#
#         [[-0.4488,  1.4382,  0.1060],
#          [ 1.3503, -0.0711,  1.5412],
#          [-0.2410,  0.0723, -0.6451],
#          [-0.3360, -0.7692,  2.2596]]], grad_fn=<EmbeddingBackward>)