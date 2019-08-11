# -*-coding:utf-8-*-

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os
import sys
sys.path.append(os.path.abspath('..'))
from config import Config
config = Config()


# Scaled Dot-Product Attention 缩放点积的实现类
class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力机制

    An attention function can be described as a query and a set of key-value pairs to an output,
    where the query, keys, values, and output are all vectors.
    The output is computed as a weighted sum of the values,
    where the weight assigned to each value is computed by a compatibility of the query with the corresponding key.
    通过确定Q和K之间的相似程度来选择V

    Attention(Q, K, V)=softmax(Q*K^T/d_k……1/2)*V

    d_k表示K的维度，默认64，当前点积得到的结果维度很大的时候，那么经过softmax函数的作用？梯度是处于很小的区域，
    这样是不利于BP的（也就是梯度消失的问题），除以一个缩放因子，能降低点积结果维度，缓解梯度消失问题

    在encoder的self-attention中，Q, K, V来自于同一个地方（相等），都是上一层encoder的输出。
    第一层的Q, K，V是word embedding和positional encoding相加得到的输入。

    在decoder的self-attention中，Q, K, V来自于同一个地方（相等），都是上一层decoder的输出。
    第一层的Q, K，V是word embedding和positional encoding相加得到的输入。
    但是在decoder中是基于以前信息来预测下一个token的，所以需要屏蔽当前时刻之后的信息，即做sequence masking

    在encoder-decoder交互的context attention层中，Q来自于decoder的上一层的输出，K和V来自于encoder的输出，K和V相同

    Q, K, V三个始终的维度是一样的，d_q=d_k=d_v
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(config.dropout)  # dropout操作
        self.softmax = nn.Softmax(dim=2)  # softmax操作，TODO 为什么是定义在dim=2

    def forward(self, q, k, v, mask):
        """
        # softmax(q*k^T/d_k^(1/2))*v
        :param q: query (B*h)*L*d
        :param k: key (B*h)*L*d
        :param v: value (B*h)*L*d
        :param mask: (B*h)*L*L
        :return:
        """
        attn = torch.bmm(q, k.transpose(1, 2))  # q*k^T (B*h)*L*L

        attn = attn / np.power(config.d_k, 0.5)  # q*k^T / d_k^(1/2)

        attn = attn.masked_fill(mask, -np.inf)  # 屏蔽掉序列补齐的位置 (B*h)*L*L

        attn = self.softmax(attn)

        attn = self.dropout(attn)

        attn = torch.bmm(attn, v)  # *v D*L*d

        return attn


# Multi-head Attention 多头注意力机制的实现类
class MultiHeadAttention(nn.Module):
    """
    multi-head attention多头注意力机制
    将Q，K，V通过线性映射（乘上权重层W），分成h(h=8)份，每一份再所缩放点积操作效果更好
    再把h份合起来，经过线性映射（乘上权重层W），得到最后的输出

    MultiHead(Q, K, V)=Concat(head_1, head_2, ..., head_h) * W^O
        head_i = Attention(Q*W_Q_i, K*W_K_i, V*W_V_i)

    每一份做缩放点积操作的d_k, d_q, d_v的维度是原来总维度的h份，
    即d_k，d_q，d_v的维度=D_k/h，D_q/h，D_v/h

    """
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

        # query, key, value的权重层
        self.w_q_s = nn.Linear(config.d_model, config.heads * config.d_q)
        self.w_k_s = nn.Linear(config.d_model, config.heads * config.d_k)
        self.w_v_s = nn.Linear(config.d_model, config.heads * config.d_v)
        # 权重层以正态分布初始化
        nn.init.normal_(self.w_q_s.weight, mean=0, std=np.sqrt(2.0 / (config.d_model + config.d_q)))
        nn.init.normal_(self.w_k_s.weight, mean=0, std=np.sqrt(2.0 / (config.d_model + config.d_k)))
        nn.init.normal_(self.w_v_s.weight, mean=0, std=np.sqrt(2.0 / (config.d_model + config.d_v)))

        self.concat_heads = nn.Linear(config.heads * config.d_v, config.d_model)  # 级联多头

        self.attention = ScaledDotProductAttention()  # 缩放点积attention计算

        self.dropout = nn.Dropout(config.dropout)

        self.layer_norm = nn.LayerNorm(config.d_model)  # LN层归一化操作

    def forward(self, query, key, value, mask):
        """

        :param q: query B*L*D
        :param k: key B*L*D
        :param v: value B*L*D
        :param mask: 屏蔽位 B*L*L
        :return: h个头的缩放点积计算结果 B*L*(h*d=d_model)
        """
        residual = query  # 残差

        batch_size, seq_len_q, dim = query.shape
        batch_size, seq_len_k, dim = key.shape
        batch_size, seq_len_v, dim = value.shape

        # query乘上权重得到一个头（但是同时计算多头） B*L*H*d
        query = self.w_q_s(query).view(batch_size, seq_len_q, config.heads, config.d_q)
        # key乘上权重得到一个头（但是同时计算多头） B*L*H*d
        key = self.w_k_s(key).view(batch_size, seq_len_k, config.heads, config.d_k)
        # value乘上权重得到一个头（但是同时计算多头） B*L*H*d
        value = self.w_v_s(value).view(batch_size, seq_len_v, config.heads, config.d_v)

        query = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_len_q, config.d_q)  # query的维度融合变换 (B*H)*L*d
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_len_k, config.d_k)  # key的维度融合变换 (B*H)*L*d
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_len_v, config.d_v)  # value的维度融合变换 (B*H)*L*d

        # 产生多头的mask
        pad_mask = mask.repeat(config.heads, 1, 1) # (B*h)*L*L

        attn = self.attention(query, key, value, pad_mask)  # 把全部的头送入scaled dot-product attn中计算

        attn = attn.view(config.heads, batch_size, seq_len_q, config.d_v)  # attn的融合变换 h*B*L*d
        attn = attn.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_len_q, -1)  # B*T*(h*d=d_model)

        attn = self.dropout(self.concat_heads(attn))  # 级联多头 B*L*(h*d=d_model)

        attn = self.layer_norm(residual + attn)  # 残差+LayerNorm

        return attn


# Position-wise Feed-Forward Networks 按位置前馈网络的实现类
class PositionWiseFeedForward(nn.Module):
    """
    按位置的前馈网络
    包含两个线性变换和一个非线性函数
    FeedForwardNet(x)=max(0, x*W_1+b_1)*W_2+b_2
    采用两个一维核大小=1的一维卷积解释 TODO 这个怎么计算的
    """
    def __init__(self):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(in_channels=config.d_model,
                             out_channels=config.d_ff,
                             kernel_size=1)  # kernel_size=1，权重w_1层

        self.w_2 = nn.Conv1d(in_channels=config.d_ff,
                             out_channels=config.d_model,
                             kernel_size=1)  # kernel_size=1，权重w_2层
        # TODO 对权重层的参数进行初始化
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.d_model)  # LN层归一化层

    def forward(self, x):
        residual = x  # 残差 B*L*(h*d=d_model)

        ffn = F.relu(self.w_1(x.transpose(1, 2)))  # max(0, x*w_1+b1) B*d_ff*L
        ffn = self.w_2(ffn)  # *w_2+b_2 B*d_ff*L

        ffn = self.dropout(ffn)
        ffn = self.layer_norm(residual + ffn.transpose(1, 2))  # 残差+LayerNorm结构 B*L*D

        return ffn