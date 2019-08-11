# -*-coding:utf-8-*-

# Transformer模型的实用工具

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import os
import sys
sys.path.append(os.path.abspath('..'))

from config import Config
config = Config()

r""" Transformer模型中涉及的各类计算工具

"""


# 位置编码类
class PositionalEncoding(nn.Module):
    r""" Positional Encoding 位置编码类说明
    由于没有使用传统的基于RNN或者CNN的结构，那么输入的序列就无法判断其顺序信息，但这序列来说是十分重要的，
    Positional Encoding的作用就是对序列中词语的位置进行编码，这样才能使模型学会顺序信息。
    使用正余弦函数编码位置，pos在偶数位置为正弦编码，奇数位置为余弦编码。
    PE(pos, 2i)=sin(pos/10000^(2i/d_model)
    PE(pos, 2i+1)=cos(pos/10000^2i/d_model)

    即给定词语位置，可以编码为d_model维的词向量，位置编码的每一个维度对应正弦曲线，
    上面表现出的是位置编码的绝对位置编码（即只能区别前后位置，不能区别前前...或者后后...的相对位置）
    又因为正余弦函数能表达相对位置信息，即:
    sin(a+b)=sin(a)*cos(b)+cos(a)*sin(b)
    cos(a+b)=cos(a)*cos(b)-sin(a)*sin(b)
    对于词汇之间的位置偏移k，PE(pos+k)可以表示成PE(pos)+PE(k)组合形式，
    那么就能表达相对位置（即能区分长距离的前后）。
    """
    def __init__(self, max_seq_len, d_model, pad_idx):
        """
        位置编码
        :param max_seq_len: 序列的最大长度
        :param d_model: 模型的维度
        :param pad_idx: 填充符位置，默认为0
        """
        super(PositionalEncoding, self).__init__()
        pos_enc = np.array([
            [pos / np.power(10000, 2.0 * (i // 2) / d_model) for i in range(d_model)]
            for pos in range(max_seq_len)])  # 构建位置编码表

        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])  # sin计算偶数位置
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])  # cos计算奇数位置

        pos_enc[pad_idx] = 0.  # 第一行默认为pad符，全部填充0

        self.pos_embedding = nn.Embedding(max_seq_len, d_model)  # 设置位置向量层 L*D
        # 载入位置编码表，并不更新位置编码层
        self.pos_embedding.from_pretrained(torch.FloatTensor(pos_enc), freeze=True)

    def forward(self, src_seq):
        # 返回该批序列每个序列的字符的位置编码embedding
        return self.pos_embedding(src_seq.to(device=config.device))


# 必须的工具类
class Tools():

    def __init__(self):
        pass

    def str_2_list(self, tgt_seq):
        """
        转化字符序列为列表序列，用于转化批数据中的字符序列
        :param tgt_seq: 输入的字符序列
        :return: 字符序列对应的列表序列
        """
        ss = []
        for s in tgt_seq:  # 把字符序列还原list序列
            s = s.lstrip('[').rstrip(']').replace(' ', '').split(',')
            ss.append([int(i) for i in s])

        return ss

    def pad_seq(self, seq, max_len):
        """
        构造带屏蔽位置符的序列
        :param seq:
        :param max_len:
        :return:
        """
        seq += [config.pad for _ in range(max_len - len(seq))]

        return seq

    def seq_2_tensor(self, seq):
        """
        将序列转化为tensor
        :param seq:
        :return:
        """
        seq = self.str_2_list(seq)  # 还原序列
        seq_max_len = max(len(s) for s in seq)  # 该批次序列中的最大长度
        # 以最大长度补齐该批次的序列并转化为tensor
        seq = Variable(torch.LongTensor([self.pad_seq(s, seq_max_len) for s in seq])).to(config.device)

        return seq

    def batch_2_tensor(self, batch_data):
        """
        把批次数据转化为tensor数据
        :param batch_data: B*L 该批次的数据
        :return:
        """
        src_seq = self.seq_2_tensor(batch_data[0])  # 生成并转化source序列
        tgt_seq = self.seq_2_tensor(batch_data[1])  # 生成并转化target序列

        return src_seq, tgt_seq

    def seq_2_pos(self, seq):
        """
        为序列构造位置索引序列
        :param seq:
        :return:
        """
        batch_size, seq_max_len = seq.shape
        pos = np.zeros((batch_size, seq_max_len))

        for idx_1, i in enumerate(seq):
            for idx_2, j in enumerate(i):
                if int(j.cpu().detach().numpy()) != 0:
                    pos[idx_1][idx_2] = idx_2 + 1
                else:
                    continue
        pos = torch.LongTensor(pos).to(device=config.device)

        return pos


# Mask屏蔽符类
class Mask():

    def __init__(self):
        pass

    # 对序列做补齐的位置
    def padding_mask(self, seq_k, seq_q):
        """
        生成padding masking TODO 待解释seq_k和seq_q的关系和来源
        :param seq_k: B*L
        :param seq_q: B*L
        :return: 产生B*T*T的pad_mask输出
        """
        seq_len = seq_q.size(1)
        pad_mask = seq_k.eq(config.pad)  # 通过比较产生pad_mask B*T

        return pad_mask.unsqueeze(1).expand(-1, seq_len, -1)

    # 不对序列做补齐的位置
    def no_padding_mask(self, seq):
        """
        pad_mask的反向操作
        :param seq: B*T
        :return: B*T*T
        """
        return seq.ne(config.pad).type(torch.float).unsqueeze(-1)

    # 序列屏蔽，用于decoder的操作中
    def sequence_mask(self, seq):
        """
        屏蔽子序列信息，防止decoder能解读到，使用一个上三角形来进行屏蔽
        seq: B*T batch_size*seq_len
        :return: seq_mask B*T*T batch_size*seq_len*seq_len
        """
        batch_size, seq_len = seq.shape
        # 上三角矩阵来屏蔽不能看到的子序列
        seq_mask = torch.triu(
             torch.ones((seq_len, seq_len), device=config.device, dtype=torch.uint8), diagonal=1)

        return seq_mask.unsqueeze(0).expand(batch_size, -1, -1)


# 损失函数计算类
class Criterion():

    def cal_loss(self, real_tgt, pre_tgt):
        """
        对模型预测输出与正确输出计算损失，并计算单词正确的个数
        :param real_tgt:
        :param pre_tgt:
        :return:
        """
        loss = F.cross_entropy(pre_tgt, real_tgt, ignore_index=config.pad, reduction=config.loss_cal)  # 计算损失

        pre = pre_tgt.max(1)[1]
        real = real_tgt.contiguous().view(-1)
        non_pad_mask = real.ne(config.pad)
        correct = pre.eq(real)
        correct = correct.masked_select(non_pad_mask).sum().item()

        return loss, correct


# 特殊学习率优化器类
class SpecialOptimizer():

    def __init__(self, optimizer, warmup_steps, d_model, step_num=0):
        """
        随着训练步骤，学习率对应改变的模型优化器
        :param optimizer: 预定义的优化器
        :param warmup_steps: 预热步
        :param d_model: 模型维度
        :param step_num: 当前的模型的的训练步数，默认从0开始
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.step_num = step_num

    # 优化器梯度清零
    def zero_grad(self):
        self.optimizer.zero_grad()

    # 优化器更新学习率和步进
    def step_update_lrate(self):
        self.step_num += 1  # 批次训练一次，即为步数一次，自加1

        # 生成当前步的学习率
        lr = np.power(self.d_model, -0.5) * np.min([np.power(self.step_num, -0.5),
                                                    np.power(self.warmup_steps, -1.5) * self.step_num])

        # 把当前步的学习率赋值给优化器
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.optimizer.step()

        return lr


# 引用他人的批数据推理测试代码
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Translator.py
class Translator(object):

    def __init__(self, model, tgt_max_len):
        self.model = model.to(config.device)

        self.word_prob_prj = nn.LogSoftmax(dim=1)  # 解码生成词的映射层

        self.model.eval()  # 设置模型为验证模型，不反向传播，更新参数

        self.tgt_max_len = tgt_max_len

    def translate_batch(self, src_seq, src_pos):
        ''' Translation work in one batch 按批次解码生成序列'''

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. 指示示例在张量中的位置'''
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            """
            Collect tensor parts associated to active instances. 收集与活动实例关联的张量部分
            :param beamed_tensor: 输入序列 [batch_size * beam_size, input_max_len]
            :param curr_active_inst_idx: 目前激活的实例索引 一维大小为batch_size的tensor
            :param n_prev_active_inst: n个之前激活的示例 batch_size
            :param n_bm: beam_size大小
            :return:
            """

            _, *d_hs = beamed_tensor.size()  # 批输入序列的最大序列长度
            n_curr_active_inst = len(curr_active_inst_idx)  # 当前的激活实例数 batch_size
            new_shape = (n_curr_active_inst * n_bm, *d_hs)  # 构造 batch_size * input_max_len的最大元祖

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)  # beamed_tensor变化的形状 batch_size * -1
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)  # 选择激活的索引
            beamed_tensor = beamed_tensor.view(*new_shape)  # 转变形状为 [batch_size * beam_size, input_max_len]

            return beamed_tensor

        def collate_active_info(
                src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list):
            """
            Sentences which are still active are collected, 收集已经完成的句子
            so the decoder will not run on completed sentences. 不会在不完成的句子上继续执行
            :param src_seq: 输入序列 [batch_size * beam_size, input_max_len]
            :param src_enc: 编码器输出 [batch_size * beam_size, input_max_len, dimension]
            :param inst_idx_to_position_map:索引到位置的匹配字典 {0:0, 1:1, ..., batch_size - 1:batch_size - 1}
            :param active_inst_idx_list: 激活的索引列表 [0, 1, 2, ..., batch_size-1]
            :return:
            """
            n_prev_active_inst = len(inst_idx_to_position_map)  # 获取字典大小 batch_size
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(config.device)  # 转化为LongTensor类型

            active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
            active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_seq, active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, n_bm):
            """
            :param inst_dec_beams: 解码的beam search对象 batch size 个
            :param len_dec_seq: 解码的序列位置，从1~最大长度+1？
            :param src_seq:输入序列  B*L
            :param enc_output:编码器的编码输出 B*L*D
            :param inst_idx_to_position_map: 索引到位置的匹配字典，{0:0，1:1,...,batch_size:batch_size}
            :param n_bm: beam search的beam size大小
            :return:
            """
            ''' Decode and update beam status, and then return active beam idx 解码并更新beam状态，然后返回激活的beam索引 '''

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                # 解码的偏置序列？是一个列表，共有batch_size个元素，每个元素是[beam，1]的矩阵，第一个值为开始符2
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                # 把解码的部分序列堆叠起来，变为[batch_size, beam_size, 1]的矩阵
                dec_partial_seq = torch.stack(dec_partial_seq).to(config.device)
                # 变换为[batch_size*beam_size, 1]的矩阵
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
                """
                准备解码序列的pos位置向量
                :param len_dec_seq: 解码序列的长度？
                :param n_active_inst: 需要激活的句子，大小为batch_size
                :param n_bm: beam_size
                :return:
                """
                # 解码部分位置向量 [1] 一个大小为[1]，值为1的tensor
                dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long, device=config.device)
                # 解码部分位置向量 扩展为其大小为[batch_size*beam_size, 1]， 每个值都是1
                dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
                return dec_partial_pos

            def predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm):
                """
                预测出来额单词
                :param dec_seq: 解码序列 [beam_size*batch_size, 1] beam_size为一个循环，第一个值为开始符2
                :param dec_pos: 解码序列为位置向量 [batch_size*batch_size, 1] 值都为1
                :param src_seq: 输入序列 [batch_size*beam_size, input_max_len]
                :param enc_output: 编码器输出 [batch_size*beam_size, input_max_lem, dimension]
                :param n_active_inst: 需要记录的序列数量 == batch_size
                :param n_bm: beam_size
                :return:
                """
                # 解码器解码输出 [batch_size*beam_size, 1，d]
                dec_output = self.model.decoder(dec_seq, src_seq, enc_output)  # 解码器输出
                # 选择最后一步的解码序列, 本来也只有一步 [batch_size*beam_size, d]
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                # 所有词进行映射输出，得到每个词的概率 [batch_size*beam_size, M]
                word_prob = F.log_softmax(self.model.word_prob_map(dec_output), dim=1)
                # 变换形状 [batch_size, beam_size, M]
                word_prob = word_prob.view(n_active_inst, n_bm, -1)

                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                """

                :param inst_beams: beam search 对象 batch_size 个
                :param word_prob: 模型预测出来的结果 [batch_size, beam_size, M]
                :param inst_idx_to_position_map: 索引到位置的匹配字典 {0:0,1:1,2:2,...,batch_size-1:batch_size-1}
                :return:
                """
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():  # 循环索引和位置
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:  # 如果没有完成
                        active_inst_idx_list += [inst_idx]

                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)  # n个激活的batch 值为batch_size

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)  # 准备beam search的目标序列
            dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)  # 准备beam search的目标序列POS位置
            word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm)

            # Update the beam with predicted word prob information and collect incomplete instances
            # 用预测的单词概率信息更新beam search并收集不完整的实例？？？
            active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():
            #-- Encode  先编码
            src_seq, src_pos = src_seq.to(config.device), src_pos.to(config.device)
            src_enc = self.model.encoder(src_seq)

            #-- Repeat data for beam search
            n_bm = config.beam_size  # beam search个数多少
            n_inst, len_s, d_h = src_enc.size()  # 批大小，该批序列最大长度，维度
            src_seq = src_seq.repeat(1, n_bm).view(n_inst * n_bm, len_s)  # 重复生成n次src_seq [n*b, l]
            src_enc = src_enc.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)  # 重复生成n次src_seq [n*b, h, d]

            #-- Prepare beams 准备beam search的对象，对象个数和batch size一致
            inst_dec_beams = [Beam(n_bm, device=config.device) for _ in range(n_inst)]

            #-- Bookkeeping for active or not  是否有效记账？？？
            active_inst_idx_list = list(range(n_inst))  # 批大小的list列表[0,2,...,batch_size]
            # 返回一个字典，{0:0,1:1,...,batch_size:batch_size}
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            #-- Decode 解码阶段
            for len_dec_seq in range(1, self.tgt_max_len + 1):  # 在1~序列最大长度+1之内循环？为什么

                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams, len_dec_seq, src_seq, src_enc, inst_idx_to_position_map, n_bm)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                src_seq, src_enc, inst_idx_to_position_map = collate_active_info(
                    src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list)

        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, config.beam_search_left)

        return batch_hyp, batch_scores


# 引用他人的beam search代码
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Beam.py
class Beam():
    ''' Beam search beam搜索'''

    def __init__(self, size, device=False):

        self.size = size
        self._done = False

        # The score for each translation on the beam.收集每一层的评分
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)  # 大小为beam size的一维tensor
        self.all_scores = []

        # The backpointers at each time-step.  # TODO 这个变量是啥意思
        self.prev_ks = []

        # The outputs at each time-step. 每一步的输出
        # 列表，里面的元素是长度为beam size，值为0的一维tensor
        self.next_ys = [torch.full((size,), config.pad, dtype=torch.long, device=device)]
        self.next_ys[0][0] = config.sos  # list第一个元素为一维大小beam size的tensor，第一个值为开始符BOS

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob):
        """
        Update beam status and check if finished or not. 更新beam状态并检查是否完成
        :param word_prob: [beam_size, M] 预测出来所有词的概率
        :return:
        """
        num_words = word_prob.size(1)  # 目标序列所有单词数

        # Sum the previous scores.  把之前所有的分数相加
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]

        flat_beam_lk = beam_lk.view(-1)  # 变换为一维矩阵，大小为 M

        # 获取其中top beam_size项的概率和索引位置
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)  # 1st sort
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)  # 2nd sort

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened as a (beam x word) array, 变换 BestScoreSid beam*word词数的数组，
        # so we need to calculate which word and beam each score came from 计算每个单词和beam 搜索分数的来源
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        # End condition is when top-of-beam is EOS.  结束条件是每个beam的顶端为EOS结束符
        if self.next_ys[-1][0].item() == config.eos:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep. 获取当前时间步的解码序列"

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)  # 初始解码序列 [beam_size, 1]，第一个为开始符
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[config.sos] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))