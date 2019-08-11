# -*-coding:utf-8-*-

r"""Transformer模型训练和验证的命令行参数程序
"""

import argparse
from torch.utils.data import DataLoader
import torch
import torch.optim as optim

from transformer.data_process import DataProcess
from transformer.utils import Criterion, SpecialOptimizer
from transformer.modules import Transformer
from transformer.model import Sequence2Sequence

import os
import sys
sys.path.append(os.path.abspath('.'))
from config import Config
config = Config()


def main():
    # 命令行参数赋值
    parser = argparse.ArgumentParser(description='Transformer模型训练&验证部分！！')

    parser.add_argument('-train_data_input', help='模型训练的输入序列集', default=os.path.abspath('.') + './data/train.en')
    parser.add_argument('-train_data_target', help='模型训练册目标序列集', default=os.path.abspath('.') + './data/train.de')
    parser.add_argument('-val_data_input', help='模型验证的输入序列集', default=os.path.abspath('.') + './data/val.en')
    parser.add_argument('-val_data_target', help='模型验证的目标序列集', default=os.path.abspath('.') + './data/val.de')

    parser.add_argument('-save_data', default='./results/words_data.pt', help='保存训练和验证的文本序列信息')

    parser.add_argument('-d_model', help='模型维度', type=int, default=512)
    parser.add_argument('-d_q', help='模型query变量的维度大小', type=int, default=64)
    parser.add_argument('-d_k', help='模型key变量的维度大小', type=int, default=64)
    parser.add_argument('-d_v', help='模型value变量的维度大小', type=int, default=64)
    parser.add_argument('-heads', help='multi-head attention 多头注意力机制中的头数', type=int, default=8)
    parser.add_argument('-layers', help='编码器和解码器堆叠的层数', type=int, default=6)
    parser.add_argument('-d_ff', help='按位置前馈层中内层的维度大小', type=int, default=2048)
    parser.add_argument('-dropout', help='模型dropout层的大小', type=int, default=0.1)

    parser.add_argument('-min_word_count', help='限制序列中单词出现的最少次数', type=int, default=5)
    parser.add_argument('-epochs', help='训练轮次', default=100)
    parser.add_argument('-train_batch_size', help='训练批次大小', type=int, default=128)
    parser.add_argument('-warm_up_step', help='训练预热步骤', type=int, default=4000)
    parser.add_argument('-beam_search', help='Beam search大小', type=int, default=5)
    parser.add_argument('-cal_loss', help='计算损失的方式', type=str, choices=['sum', 'mean'], default='sum')

    parser.add_argument('-gpu_or_not', help='是否使用GPU运行程序', default=True)
    parser.add_argument('-all_gpu', help='是否使用全部的GPU运行程序', type=str, choices=['all', 'one'], default='all')
    parser.add_argument('-label_smooth', help='是否使用标签光滑选项', default=True)
    parser.add_argument('-save_trained_model', help='是否保存训练好的模型', default=True)
    parser.add_argument('-save_trained_model_type', help='是否保存训练好的模型', type=str, choices=['all', 'best'],
                        default='best')

    args = parser.parse_args()  # 赋值参数
    config.args_2_variable_train(args)  # 修改默认参数

    # 准备数据
    data_obj = DataProcess(train_src=args.train_data_input,
                           train_tgt=args.train_data_target,
                           val_src=args.val_data_input,
                           val_tgt=args.val_data_target)

    src_tgt, src_lang, tgt_lang = data_obj.get_src_tgt_data()
    *_, src_tgt_seq_train = data_obj.word_2_index(
        args.train_data_input, args.train_data_target, src_lang, tgt_lang)  # 训练数据
    *_, src_tgt_seq_val = data_obj.word_2_index(
        args.val_data_input, args.val_data_target, src_lang, tgt_lang)  # 验证数据

    # 保存数据
    words_data = {
        'src_lang':{
            'name': src_lang.name,
            'trimmed': src_lang.trimmed,
            'word2index': src_lang.word2index,
            'word2count': src_lang.word2count,
            'index2word': src_lang.index2word,
            'n_words': src_lang.n_words,
            'seq_max_len': src_lang.seq_max_len},
        'tgt_lang':{
            'name': tgt_lang.name,
            'trimmed': tgt_lang.trimmed,
            'word2index': tgt_lang.word2index,
            'word2count': tgt_lang.word2count,
            'index2word': tgt_lang.index2word,
            'n_words': tgt_lang.n_words,
            'seq_max_len': tgt_lang.seq_max_len},
        'data_obj':{
            'src_max_len': data_obj.src_max_len,
            'tgt_max_len': data_obj.tgt_max_len}}
    torch.save(words_data, args.save_data)  # 保存批次数据到本地

    # 打包成批次数据
    train_data_loader = DataLoader(src_tgt_seq_train, config.batch_size, True, drop_last=False)
    val_data_loader = DataLoader(src_tgt_seq_val, config.batch_size, False, drop_last=False)

    # 定义transformer模型
    transformer = Transformer(input_vocab_num=src_lang.n_words,
                              target_vocab_num=tgt_lang.n_words,
                              src_max_len=data_obj.src_max_len,
                              tgt_max_len=data_obj.tgt_max_len).to(config.device)

    # 定义优化器
    optimizer = SpecialOptimizer(
        optimizer=torch.optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()), betas=(0.9, 0.98), eps=1e-09),
        warmup_steps=config.warmup_step,
        d_model=config.d_model)

    # 定义损失函数
    criterion = Criterion()

    # 定义计算模型
    seq2seq = Sequence2Sequence(transformer=transformer, optimizer=optimizer, criterion=criterion)

    # 训练&验证
    seq2seq.train_val(train_data_loader, val_data_loader)


if __name__ == '__main__':
    main()