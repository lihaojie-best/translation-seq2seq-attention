import torch
from torch import nn

import config


class TranslationEncoder(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        """
        encode 只有两层 一层是 embedding 层 一层是循环神经网络层
        """
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_idx)
        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.ENCODE_HIDDEN_SIZE,
                          batch_first=True,
                          num_layers=config.ENCODER_LAYERS,
                          bidirectional=True)

    def forward(self, x):
        # x shape [batch_size, seq_len]
        embed = self.embedding(x)  # embed shape [batch_size,seq_len,embedding_dim]
        output, hidden = self.gru(embed)  # 解码器的输入是 编码器的 hidden
        # hidden shape [num_layers * num_directions, batch_size, hidden_size]
        last_layer_forward_hidden = hidden[-2]
        last_layer_backword_hidden = hidden[-1]
        # 将隐藏状态 在最后一维度拼接起来 [ [1, 2, 3],[4, 5, 6] ] 于[[10, 20, 30, 40],[50, 60, 70, 80] ]最后一维度拼接
        # 为[
        #     [ 1,  2,  3, 10, 20, 30, 40],
        #     [ 4,  5,  6, 50, 60, 70, 80]
        # ]
        context_vector = torch.cat((last_layer_forward_hidden, last_layer_backword_hidden), dim=-1)
        return context_vector
class TranslationDecoder(nn.Module):
    def __init__(self,vocab_size,padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_idx)
        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.DECODE_HIDDEN_SIZE,
                          batch_first=True)
        self.linear = nn.Linear(in_features=config.DECODE_HIDDEN_SIZE,
                                out_features=vocab_size)
    def forward(self,x,context_vector):
        # x.shape [batch_size, seq_len=1] 因为decode层需要自回归 所以是一个词处理完 再把这个词给自己 接着处理
        embed = self.embedding(x)
        # embed.shape [batch_size, seq_len=1, embedding_dim]
        output,hidden, = self.gru(embed, context_vector)
        # output.shape [batch_size, seq_len=1, hidden_size]
        # hidden.shape [num_layers*num_direction = 1, batch_size, hidden_size]
        output = self.linear(output)
        # output.shape [batch_size, seq_len=1, vocab_size] # 全连接层 将输出映射到vocab_size维度
        return output,hidden
