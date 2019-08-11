# -*-coding:utf-8-*-

r""" Transformer模型的推理测试部分命令行程序

"""

import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from transformer.utils import Translator, Tools
from transformer.modules import Transformer
from transformer.data_process import DataProcess

import os
import sys
sys.path.append(os.path.abspath('.'))
from config import Config
config = Config()

def main():
    print('\n===开始推理测试===\n\n')

    def index_2_word(lang, seq):
        """ 转化索引到单词"""
        seq = [int(idx.detach()) for idx in seq]
        new_seq = []
        for i in seq:
            if i != config.sos and i != config.eos and i != config.pad:
                new_seq.append(i)
        idx_2_word = [lang['index2word'][i] for i in new_seq]

        return idx_2_word

    # 命令行参数
    parser = argparse.ArgumentParser(description='Transformer模型推理测试部分！')

    parser.add_argument('-save_model', default='./results/best_model.chkpt', help='训练好的模型的存放的路径')
    parser.add_argument('-save_data', default='./results/words_data.pt', help='保存训练和验证的文本序列信息')
    parser.add_argument('-infer_data_input', default='./data/test.en', help='推理测试输入数据集的路径')
    parser.add_argument('-infer_data_target', default='./data/test.de', help='推理测试目标数据集的路径')
    parser.add_argument('-pre_target', default='./results/input_target_infer.txt', help='推理预测结果存放的路径')
    parser.add_argument('-infer_batch_size', default=32, type=int, help='推理预测阶段的批大小')
    parser.add_argument('-beam_search_size', default=5, type=int, help='Beam search搜索的宽度')
    parser.add_argument('-infer_n_best', default=1, type=int, help='通过Beam search后，推理预测出top n的句子')

    args = parser.parse_args()  # 赋值参数
    config.args_2_variable_infer(args)  # 修改默认参数

    # 加载训练&验证字符信息
    words_data = torch.load(args.save_data)
    source_lang = words_data['src_lang']
    target_lang = words_data['tgt_lang']
    data_obj = words_data['data_obj']

    checkpoint = torch.load(args.save_model, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    transformer = Transformer(
        input_vocab_num=source_lang['n_words'],
        target_vocab_num=target_lang['n_words'],
        src_max_len=data_obj['src_max_len'],
        tgt_max_len=data_obj['tgt_max_len'])

    transformer.load_state_dict(checkpoint['model'])
    print('加载预训练的模型参数完成！')

    infer = Translator(model=transformer, tgt_max_len=data_obj['tgt_max_len'])  # 推理预测模型

    data_obj = DataProcess()
    *_, src_tgt_seq = data_obj.word_2_index(
        args.infer_data_input, args.infer_data_target, source_lang, target_lang)  # 测试数据
    # 打包批次数据
    data_loader = DataLoader(dataset=src_tgt_seq, batch_size=args.infer_batch_size, shuffle=True, drop_last=False)

    with open(args.pre_target, 'w', encoding='utf-8') as f:
        for batch_dat in tqdm(data_loader, desc='Inferring...', leave=True):  # 迭代推理批次数据
            src_seq, tgt_seq = Tools().batch_2_tensor(batch_dat)  # 获得输入序列和实际目标序列
            src_pos = Tools().seq_2_pos(src_seq)  # 得到输入序列的pos位置向量
            all_pre_seq, all_pre_seq_p = infer.translate_batch(src_seq, src_pos)  # 获得所有预测的结果和对应的概率

            for index, pre_seq in enumerate(all_pre_seq):
                src_word_seq = index_2_word(source_lang, src_seq[index])  # 清洗输入序列并转化为字符
                tgt_word_seq = index_2_word(target_lang, tgt_seq[index])  # 清洗目标序列并转化为字符
                for seq in pre_seq:  # 清洗预测序列并转化为字符
                    new_seq = []
                    for i in seq:
                        if i != config.sos and i != config.eos and i != config.pad:
                            new_seq.append(i)
                    pre_word_seq = [target_lang['index2word'][idx] for idx in new_seq]

                f.write('输入序列->：' + ' '.join(src_word_seq) + '\n')  # 写入输入序列
                f.write('->预测序列：' + ' '.join(pre_word_seq) + '\n')  # 写入预测序列
                f.write('==目标序列：' + ' '.join(tgt_word_seq) + '\n\n')  # 写入实际序列

    print('推理预测序列完毕！')


if __name__ == '__main__':
    main()
