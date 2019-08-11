# -*-coding:utf-8-*-

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import DataParallel

from transformer.data_process import DataProcess
from transformer.utils import SpecialOptimizer, Criterion
from transformer.model import Sequence2Sequence
from transformer.modules import Transformer

import os
import sys
sys.path.append(os.path.abspath('.'))

from config import Config
config = Config()

r"""Transformer模型训练，验证和测试的主函数。
    该主函数不通过命令行输入参数，所有参数可以在config.py中修改添加。
    该代码直接运行训练&验证，然后进行推理测试，不会保存中间训练的结果文件。
    如果你希望能保存中间训练的模型参数，请执行先train_val.py训练（验证）模型和然后执行infer.py文件推理测试。
"""


def main(train_src, train_tgt, val_src, val_tgt, test_src, test_tgt):
    """
    transformer模型运行的主函数，需要给出训练，验证和测试的数据集
    :param train_src:
    :param train_tgt:
    :param val_src:
    :param val_tgt:
    :param test_src:
    :param test_tgt:
    :return: 模型推理生成的结果文件
    """
    # 准备数据
    data_obj = DataProcess(train_src, train_tgt, val_src, val_tgt)  # 数据对象
    src_tgt, src_lang, tgt_lang = data_obj.get_src_tgt_data()
    *_, src_tgt_seq_train = data_obj.word_2_index(train_src, train_tgt, src_lang, tgt_lang)  # 训练数据
    *_, src_tgt_seq_val = data_obj.word_2_index(val_src, val_tgt, src_lang, tgt_lang)  # 验证数据
    *_, src_tgt_seq_test = data_obj.word_2_index(test_src, test_tgt, src_lang, tgt_lang)  # 测试数据

    # 打包成批次数据
    train_data_loader = DataLoader(src_tgt_seq_train, config.batch_size, True, drop_last=False)
    val_data_loader = DataLoader(src_tgt_seq_val, config.batch_size, False, drop_last=False)
    test_data_loader = DataLoader(src_tgt_seq_test, config.infer_batch, True, drop_last=False)

    # 定义transformer模型
    transformer = Transformer(input_vocab_num=src_lang.n_words,
                              target_vocab_num=tgt_lang.n_words,
                              src_max_len=data_obj.src_max_len,
                              tgt_max_len=data_obj.tgt_max_len).to(config.device)

    if config.multi_gpu:
        transformer = DataParallel(transformer)  # 多GPU运行程序  TODO 尚未修改完成，请勿使用！

    # 定义优化器
    optimizer = SpecialOptimizer(
        optimizer=torch.optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()), betas=(0.9, 0.98), eps=1e-09),
        warmup_steps=config.warmup_step,
        d_model=config.d_model)

    # 定义损失函数
    criterion = Criterion()

    # 定义计算模型
    seq2seq = Sequence2Sequence(transformer= transformer, optimizer=optimizer, criterion=criterion)

    # 训练&验证
    # seq2seq.train_val(train_data_loader, val_data_loader)

    # 推理测试
    seq2seq.infer(test_data_loader, src_lang, tgt_lang, data_obj.tgt_max_len)


if __name__ == '__main__':
    # 训练数据集
    train_source = os.path.abspath('.') + '/data/train.en'
    train_target = os.path.abspath('.') + '/data/train.de'

    # 验证数据集
    val_source = os.path.abspath('.') + '/data/val.en'
    val_target = os.path.abspath('.') + '/data/val.de'

    # 测试数据集
    test_source = os.path.abspath('.') + '/data/test.en'
    test_target = os.path.abspath('.') + '/data/test.de'

    main(train_source, train_target, val_source, val_target, test_source, test_target)
