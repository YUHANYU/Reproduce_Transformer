# -*-coding:utf-8-*-

import torch.nn as nn

import os
import sys
sys.path.append(os.path.abspath('..'))

from config import Config
config = Config()

from transformer.sublayers import MultiHeadAttention, PositionWiseFeedForward

# Encoder的一层
class EncoderLayer(nn.Module):
    """
    encoder的一层由两个子层构成，multi-head attention+FFN
    """
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.multi_heads_attn = MultiHeadAttention()  # 编码器子层一，多头注意力机制层
        self.feed_forward = PositionWiseFeedForward()  # 编码器子层二，按位置的前馈层

    def forward(self, enc_in, pad_mask, no_pad_mask):
        """
        :param enc_in: 输入批次序列 B*L
        :param pad_mask: 屏蔽位B*L*L
        :param no_pad_mask: 屏蔽位B*L*1
        :return: B*L*(h*d=d_model)
        """
        attn = self.multi_heads_attn(enc_in, enc_in, enc_in, pad_mask)  # 子层一，多头注意力层计算 B*L*(h*d=d_model)
        attn *= no_pad_mask  # 进一步屏蔽掉补齐位 B*L*(h*d=d_model)

        output = self.feed_forward(attn)  # 子层二，按位置前馈层计算 B*L*(h*d=d_model)
        output *= no_pad_mask  # 进一步屏蔽掉补齐位 B*L*(h*d=d_model)

        return output


# Decoder的一层
class DecoderLayer(nn.Module):
    """
    decoder的一层三个子层构成，mask multi-head attention + multi-head attention + FFN
    """
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.mask_multi_head_attn = MultiHeadAttention()  # 解码器子层一，带mask的多头注意力机制层
        self.multi_head_attn = MultiHeadAttention()  # 解码器子层二，多头注意力机制层
        self.feed_forward = PositionWiseFeedForward()  # 解码器子层三，按位置前馈层

    def forward(self, dec_in, enc_out, no_mask, pad_seq_mask, dec_enc_mask):
        """

        :param dec_in: 目标批序列 B*
        :param enc_out: 编码输出 B*T*?
        :param no_mask: 序列补齐位的屏蔽位 B*T*1
        :param pad_seq_mask: 序列补齐位的屏蔽位 B*T*T
        :param dec_enc_mask: 编码-解码屏蔽位 B*T*T
        :return: 一层解码计算的结果 B*T*(d*h=d_model)
        """
        attn = self.mask_multi_head_attn(dec_in, dec_in, dec_in, pad_seq_mask)  # 子层一，mask多头注意力层计算 B*L*(d*h)
        attn = attn * no_mask if no_mask is not None else attn  # 屏蔽序列补齐位 B*L*(d*h)

        attn = self.multi_head_attn(attn, enc_out, enc_out, dec_enc_mask)  # 子层二，多头（交互）注意力层计算 B*L*(d*h)
        attn = attn * no_mask if no_mask is not None else attn  # 屏蔽序列补齐位 B*L*(d*h)

        out = self.feed_forward(attn)  # 子层三，按位置前馈层计算 B*L*(d*h)
        out = out * no_mask if no_mask is not None else out  # 屏蔽序列补齐位 B*L*(d*h)

        return out