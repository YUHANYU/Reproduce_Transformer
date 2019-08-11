# -*-coding_utf-8-*-

import torch.nn as nn

import os
import sys
sys.path.append(os.path.abspath('..'))

from config import Config
config = Config()

from transformer.utils import PositionalEncoding, Mask, Tools
from transformer.layers import EncoderLayer, DecoderLayer


# Encoder的计算
class Encoder(nn.Module):
    """
    Encoder由6层EncodeLayer堆叠构成
    """
    def __init__(self, input_vocab_num, max_seq_len, pad_idx=0):
        """
        :param input_vocab_num: 全部输入序列的词典的单词数
        :param max_seq_len: 输入序列最大长度
        :param pad_idx: pad的填充位置，默认为0
        """
        super(Encoder, self).__init__()
        self.word_embedding = nn.Embedding(input_vocab_num, config.d_model, padding_idx=pad_idx)  # 词向量层 N*D
        self.pos_encoding = PositionalEncoding(max_seq_len + 1, config.d_model, pad_idx)  # 位置向量层 (N+1)*D
        self.encoder_layers = nn.ModuleList([EncoderLayer() for _ in range(config.layers)])  # 堆叠n层encoder_layer

        self.pad_obj = Mask()  # mask对象
        self.tool = Tools()  # 工具对象

    def forward(self, src_seq):
        """

        :param src_seq: 输入批序列 B*L
        :return: 6层encoder子层的计算结果 B*L*？
        """
        src_pos = self.tool.seq_2_pos(src_seq)  # 生成输入序列对应的位置序列 B*L
        # Encoder第一层的输入是词向量word embedding + 位置向量positional encoding B*L*D
        enc_in = self.word_embedding(src_seq) + self.pos_encoding(src_pos)

        pad_mask = self.pad_obj.padding_mask(src_seq, src_seq)  # pad_mask 由补齐序列产生的屏蔽位 B*L*L
        no_pad_mask = self.pad_obj.no_padding_mask(src_seq)  # 序列补齐位的屏蔽位 B*L*1

        enc_out = 0
        for encoder_layer in self.encoder_layers:  # 循环计算每一层
            enc_out = encoder_layer(enc_in, pad_mask, no_pad_mask)
            enc_in = enc_out  # 上一层的输出等于下一层的输入

        return enc_out


# Decoder的计算
class Decoder(nn.Module):
    """
    Decoder由6层DecoderLayer构成
    """
    def __init__(self, target_vocab_num, max_seq_len, pad_idx=0):
        """

        :param target_vocab_num: 全部目标序列的单词数
        :param max_seq_len: 全部目标序列的最大长度
        :param pad_idx: 屏蔽位，默认为0
        """
        super(Decoder, self).__init__()
        self.word_embedding = nn.Embedding(target_vocab_num, config.d_model)  # 构建词向量层 M*D
        self.pos_encoding = PositionalEncoding(max_seq_len + 1, config.d_model, pad_idx)  # 构建位置向量层 (M+1)*D
        self.decoder_layers = nn.ModuleList([DecoderLayer() for _ in range(config.layers)])  # 堆叠n层DecoderLayer

        self.mask_obj = Mask()  # mask类
        self.tool = Tools()  # 工具类

    def forward(self, tgt_seq, src_seq, enc_out):
        """

        :param tgt_seq: 该批目标序列 B*L
        :param src_seq: 该批输入序列 B*L
        :param enc_out: 编码器的输出 B*L*(d*h=d_model)
        :return: 解码器的输出 B*L*(d*h)
        """
        tgt_pos = self.tool.seq_2_pos(tgt_seq)  # 生成目标序列的位置向量 B*L
        no_pad_mask = self.mask_obj.no_padding_mask(tgt_seq)  # 生成target序列补齐位的屏蔽位 B*L*1

        pad_mask = self.mask_obj.padding_mask(tgt_seq, tgt_seq)  # 生成序列的补齐位的屏蔽位 B*L*L
        seq_mask = self.mask_obj.sequence_mask(tgt_seq)  # 生成子序列屏蔽位（上三角形） B*L*L
        pad_seq_mask = (pad_mask + seq_mask).gt(0)  # 在解码器中，结合两种mask B*L*L

        # 在第二层的多头注意力机制中，产生context类的mask B * tgt_L * src_L
        dec_enc_mask = self.mask_obj.padding_mask(src_seq, tgt_seq)

        # Decoder的第一层为词向量word embedding+位置向量 embedding
        dec_in = self.word_embedding(tgt_seq) + self.pos_encoding(tgt_pos)  # B*L*(h*d=d_model)

        dec_out = 0
        for decoder_layer in self.decoder_layers:  # 循环计算每一层
            dec_out = decoder_layer(dec_in, enc_out, no_pad_mask, pad_seq_mask, dec_enc_mask)
            dec_in = dec_out  # 上一层的输出等于下一层的输入

        return dec_out


# Transformer的实现类
class Transformer(nn.Module):
    """
    Transformer由Encoder和Decoder构成
    """
    def __init__(self, input_vocab_num, target_vocab_num, src_max_len, tgt_max_len):
        """
        Transformer模型的主类
        :param input_vocab_num: 全部输入序列的词典的单词数
        :param target_vocab_num: 全部目标序列的词典的单词数
        :param src_max_len: 全部输入序列的最大长度
        :param tgt_max_len: 全部目标序列的最大长度
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(input_vocab_num, src_max_len)
        self.decoder = Decoder(target_vocab_num, tgt_max_len)
        self.word_prob_map = nn.Linear(config.d_model, target_vocab_num, bias=False)  # 最后的线性转换层 D*M
        nn.init.xavier_normal_(self.word_prob_map.weight)  # 初始化线性层权重分头

    def forward(self, src_seq, tgt_seq):
        """

        :param src_seq: 输入批次序列 B*L
        :param tgt_seq: 目标批次序列 B*L
        :return: 映射到目标序列单词表的输出 B*L*M
        """

        tgt_seq = tgt_seq[:, :-1]

        enc_out = self.encoder(src_seq)  # 编码器编码输入序列 B*L*(h*d=d_model)

        dec_out = self.decoder(tgt_seq, src_seq, enc_out)  # 解码器解码输入序列（训练时也计算目标序列）

        dec_out = self.word_prob_map(dec_out)  # 映射到全部的单词 B*L*M

        output = dec_out  # 使用的是交叉熵，那就不需要做softmax了

        pre = output.view(-1, output.size(2))  # 维度变换，为了下一步的loss的计算 (B*L)*M

        return pre