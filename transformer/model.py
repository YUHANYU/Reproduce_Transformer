# -*-coding:utf-8-*-

import torch

import math
import datetime
from tqdm import tqdm
import numpy as np

from visdom import Visdom
# from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

from transformer.utils import Tools, Translator
from transformer.modules import Transformer

import os
import sys
sys.path.append(os.path.abspath('..'))

from config import Config
config = Config()

from visual import Visualization
pic = Visualization()

r""" Transformer模型的训练，验证和测试类
"""

# 总的计算模型，序列到序列
class Sequence2Sequence():

    def __init__(self, transformer, optimizer, criterion):
        self.transformer = transformer  # transformer模型
        self.optimizer = optimizer  # 优化器
        self.criterion = criterion  # 损失函数

        self.tool = Tools()  # 工具类

        self.val_acc_all = [0]  # 收集验证集所有的准确率
        self.save_checkpoint = 0  # 模型最佳保存点的次数

        self.val_all_batch = 0
        self.train_all_batch = 0

        self.train_loss = []
        self.val_loss = []
        self.train_ppl = []
        self.val_ppl = []
        self.train_acc = []
        self.val_acc = []

        if config.visual and config.use_visdom: # Visdom可视化
            # 损失初始化
            viz_loss = Visdom(env='Transformer_Train_Val_Loss')
            self.viz_loss = viz_loss
            t_loss_x, t_loss_y = 0, 0
            win_loss_1 = viz_loss.line(np.array([t_loss_x]), np.array([t_loss_y]),
                                       opts=dict(title='训练loss', legend=['train'], markers=True, markersize=5))

            v_loss_x, v_loss_y = 0, 0
            win_loss_2 = viz_loss.line(np.array([v_loss_x]), np.array([v_loss_y]),
                                       opts=dict(title='验证loss', legend=['val'], markers=True, markersize=5))
            self.win_loss_1 = win_loss_1
            self.win_loss_2 = win_loss_2

            # ppl点初始化
            viz_ppl = Visdom(env='Transformer_Train_Val_PPL')
            self.viz_ppl = viz_ppl
            t_ppl_x, t_ppl_y = 0, 0
            win_ppl_1 = viz_ppl.line(np.array([t_ppl_x]), np.array([t_ppl_y]),
                                     opts=dict(title='训练PPL', legend=['train'], markers=True, markersize=5))

            v_ppl_x, v_ppl_y = 0, 0
            win_ppl_2 = viz_ppl.line(np.array([v_ppl_x]), np.array([v_ppl_y]),
                                     opts=dict(title='验证PPL', legend=['val'], markers=True, markersize=5))
            self.win_ppl_1 = win_ppl_1
            self.win_ppl_2 = win_ppl_2

            # acc点初始化
            viz_acc = Visdom(env='Transformer_Train_Val_ACC')
            self.viz_acc = viz_acc
            t_acc_x, t_acc_y = 0, 0
            win_acc_1 = viz_acc.line(np.array([t_acc_x]), np.array([t_acc_y]),
                                     opts=dict(title='验证ACC', legend=['train'], markers=True, markersize=5))
            v_acc_x, v_acc_y = 0, 0

            win_acc_2 = viz_acc.line(np.array([v_acc_x]), np.array([v_acc_y]),
                                     opts=dict(title='验证ACC', legend=['val'], markers=True, markersize=5))
            self.win_acc_1 = win_acc_1
            self.win_acc_2 = win_acc_2

    # 训练&验证
    def train_val(self, train_loader, val_loader):
        print('\n===开始训练&验证===\n')

        train_log = None
        val_log = None
        if config.log:  # 模型写入日志
            train_log = open((os.path.abspath('.') + '/results/train.log'), 'w', encoding='utf-8')
            train_log.write(
                '轮次：{epoch:3.0f}, '
                '当前批次：{step:3.0f}, '
                '累计批次：{total_step:7.0f}, '
                '批大小：{batch_size:3.0f}, '
                '损失：{loss:10.6f}, '
                '该批学习率：{lr:15.10f}\n\n'
                .format(epoch=0, step=0, total_step=0, batch_size=0, loss=0, lr=0))

            val_log = open((os.path.abspath('.') + '/results/val.log'), 'w', encoding='utf-8')
            val_log.write(
                '轮次：{epoch:3.0f}, '
                '批大小：{batch_size:3.0f}, '
                '损失：{loss:10.6f}\n\n, '
                .format(epoch=0, batch_size=0, loss=0))

        for epoch in range(config.epochs):  # 按轮次训练数据
            print('[训练轮次 {}]'.format(epoch))

            step = 0  # 每个轮次的批次数

            each_batch_loss = []  # 记录每一批次数据的损失
            each_batch_lr = []  # 记录每一个批次的学习率
            self.transformer.train()  # 设置模型为训练状态
            total_loss = 0  # 本轮次所有批次数据的训练损失
            total_word_num = 0  # 本轮次所有批次数据的词数
            total_word_correct = 1  # 本轮次所有批次数据中单词正确的个数
            epoch_start_time = datetime.datetime.now()  # 一个轮次模型的计算开始时间

            for batch_data in tqdm(train_loader, mininterval=1, desc='Training...', leave=False):  # 迭代计算批次数据
                self.train_all_batch += 1  # 总批次加一
                step += 1  # 该轮次批数加一

                self.optimizer.zero_grad()  # 优化器梯度清零
                src_seq, tgt_seq = self.tool.batch_2_tensor(batch_data)  # 得到输入和目标序列 B*L B*L
                pre_tgt = self.transformer(src_seq, tgt_seq)  # transformer模型预测结果
                real_tgt = tgt_seq[:, 1:].contiguous().view(-1)  # 构造实际的目标序列token
                loss, correct = self.criterion.cal_loss(real_tgt, pre_tgt)
                loss.backward(retain_graph=True)  # 损失反向传播(计算损失后还保留变量)
                learn_rate = self.optimizer.step_update_lrate()  # 更新学习率，优化器步进

                total_loss += loss.item()  # 累加损失
                each_batch_loss.append(loss.detach())  # 获取该批次损失
                each_batch_lr.append(learn_rate)

                non_pad_mask = real_tgt.ne(config.pad)
                word_num = non_pad_mask.sum().item()
                total_word_num += word_num
                total_word_correct += correct

                # 写入训练日志
                if train_log:
                    train_log.write(
                        '轮次：{epoch:3.0f}, '
                        '当前批次：{step:3.0f}, '
                        '累计批次：{total_step:7.0f}, '
                        '批大小：{batch_size:3.0f}, '
                        '损失：{loss:10.6f}, '
                        '该批学习率：{lr:15.10f}\n'
                        .format(epoch=epoch, step=step, total_step=self.train_all_batch,
                                batch_size=len(batch_data[0]), loss=loss.detach(), lr=learn_rate))

                if config.visual and config.use_visdom:
                    self.viz_loss.line(np.array([loss.cpu().detach()]), np.array([self.train_all_batch]),
                                       win=self.win_loss_1, update='append')

                self.train_loss.append(loss.cpu().detach())

            loss_per_word = total_loss / total_word_num  # 平均到每个单词的损失
            ppl = math.exp(min(loss_per_word, 100))  # 困惑度，越小越好
            acc = total_word_correct / total_word_num  # 平均到每个单词的准确率acc
            acc = 100 * acc  # 准确率，越大越好
            epoch_end_time = datetime.datetime.now()
            self.train_ppl.append(ppl)
            self.train_acc.append(acc)

            if config.visual and config.use_visdom:
                self.viz_ppl.line(np.array([ppl]), np.array([epoch]), win=self.win_ppl_1, update='append')
                self.viz_acc.line(np.array([acc]), np.array([epoch]), win=self.win_acc_1, update='append')

            print('批数 %4.0f' % step,
                  '| 累积批数 %8.0f' % self.train_all_batch,
                  '| 批大小 %3.0f' % config.batch_size,
                  '| 耗时', epoch_end_time - epoch_start_time)

            print('训练',
                  '| 困惑度PPL↓ %10.6f' % ppl,
                  '| 准确率ACC↑ %10.5f' % acc,
                  '| 首批损失 %10.5f' % each_batch_loss[0],
                  '| 尾批损失 %10.5f' % each_batch_loss[-2])  # 最后一批不满完整batch_size，不显示

            train_log.write('困惑度PPL↓：{PPL:10.5f}, 准确率ACC↑：{ACC:10.5f}\n\n'.format(PPL=ppl, ACC=acc))

            self.evaluate(val_loader, epoch, self.train_all_batch, val_log)  # 每一个训练轮次结束，验证一次

        pic.train_loss(self.train_loss)
        pic.val_loss(self.val_loss)
        pic.ppl(self.train_ppl, self.val_ppl)
        pic.acc(self.train_acc, self.val_acc)
        # TODO 训练结束，待保存训练模型
        print('训练&验证结束！')

    # 每个轮次训练结束，用验证数据验证模型
    def evaluate(self, data_loader, epoch, batch, val_log):
        self.transformer.eval()  # 设置模型为验证状态
        total_loss = 0  # 本轮次所有批次数据的训练损失
        total_word_num = 0  # 本轮次所有批次数据的词数
        total_word_correct = 0  # 本轮次所有批次数据中单词正取的个数

        with torch.no_grad():  # 设置验证产生的损失不更新模型
            for batch_data in tqdm(data_loader, mininterval=1, desc='Validating...', leave=False):  # 迭代计算批次数据
                src_seq, tgt_seq = self.tool.batch_2_tensor(batch_data)  # 获取输入序列和验证序列
                pre_tgt = self.transformer(src_seq, tgt_seq)  # 模型预测的目标序列
                real_tgt = tgt_seq[:, 1:].contiguous().view(-1)  # 构建真实的目标序列
                loss, correct = self.criterion.cal_loss(real_tgt, pre_tgt)

                total_loss += loss.item()  # 累加损失
                self.val_all_batch += 1  # 累加该轮次验证批次数

                non_pad_mask = real_tgt.ne(config.pad)
                word_num = non_pad_mask.sum().item()
                total_word_num += word_num
                total_word_correct += correct

                if val_log:
                    val_log.write(
                        '轮次：{epoch:3.0f}, '
                        '批大小：{batch_size:3.0f}, '
                        '损失：{loss:10.6f}\n'
                         .format(epoch=epoch, batch_size=len(batch_data[0]), loss=loss.detach()))

                if config.visual and config.use_visdom:  # Visdom 可视化
                    self.viz_loss.line(np.array([loss.cpu().detach()]), np.array([self.val_all_batch]),
                                       win=self.win_loss_2, update='append')

                self.val_loss.append(loss.cpu().detach())

            loss_per_word = total_loss / total_word_num  # 平均到每个单词的损失
            ppl = math.exp(min(loss_per_word, 100))  # 困惑度，越小越好
            acc = total_word_correct / total_word_num  # 平均到每个单词的准确率acc
            acc = 100 * acc  # 准确率，越大越好

            self.val_acc_all.append(acc)  # 收集每一次验证集的准确率
            self.val_ppl.append(ppl)
            self.val_acc.append(acc)

            if config.visual and config.use_visdom:
                self.viz_ppl.line(np.array([ppl]), np.array([epoch]), win=self.win_ppl_2, update='append')
                self.viz_acc.line(np.array([acc]), np.array([epoch]), win=self.win_acc_2, update='append')

            print('验证',
                  '| 困惑度PPL↓ %10.5f' % ppl,
                  '| 准确率ACC↑ %10.5f' % acc)

            if val_log:
                val_log.write('困惑度PPL↓：{PPL:10.5f}, 准确率ACC↑：{ACC:10.5f}\n\n'.format(PPL=ppl, ACC=acc))

        # 模型保存点
        if config.save_trained_model:
            model_state_dict = self.transformer.state_dict()  # 保存训练模型状态
            checkpoint = {  # 保存点的信息
                'model': model_state_dict,
                'settings': config,
                'epoch': epoch,
                'total batch': batch}

            if config.save_trained_model_type == 'all':
                model_name = os.path.abspath('.') + '/all' + '_acc_{acc:3.3f}.chkpt'.format(acc=100*acc)
                torch.save(checkpoint, model_name)
            elif config.save_trained_model_type == 'best':
                model_name = os.path.abspath('.') + '/results/best_model.chkpt'
                if acc >= max(self.val_acc_all):
                    torch.save(checkpoint, model_name)
                    self.save_checkpoint += 1
                    print('已经第{}次更新模型最佳保存点！'.format(self.save_checkpoint))

    # 批序列推理
    def infer(self, data_loader, source_lang, target_lang, tgt_max_len):
        print('\n===开始推理测试===\n\n')

        def index_2_word(lang, seq):
            """ 转化索引到单词"""
            seq = [int(idx.detach()) for idx in seq]
            new_seq = []
            for i in seq:
                if i != config.sos and i != config.eos and i != config.pad:
                    new_seq.append(i)
            if type(lang) != dict:
                idx_2_word = [lang.index2word[i] for i in new_seq]
            else:
                idx_2_word = [lang['index2word'][i] for i in new_seq]

            return idx_2_word

        if config.save_trained_model:  # 如果有预训练好的模型保存点
            # 加载训练&验证字符信息
            words_data = torch.load(config.save_data)
            source_lang = words_data['src_lang']
            target_lang = words_data['tgt_lang']
            data_obj = words_data['data_obj']

            checkpoint = torch.load(config.checkpoint, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            transformer = Transformer(
                input_vocab_num=source_lang['n_words'],
                target_vocab_num=target_lang['n_words'],
                src_max_len=data_obj['src_max_len'],
                tgt_max_len=data_obj['tgt_max_len'])

            transformer.load_state_dict(checkpoint['model'])
            self.transformer = transformer

            print('加载预训练的模型参数完成！')

        infer = Translator(self.transformer, tgt_max_len)  # 批次数据推理类

        with open((os.path.abspath('.') + '/results/input_target_infer.txt'), 'w', encoding='utf-8') as f:
            for batch_dat in tqdm(data_loader, desc='Inferring...', leave=False):  # 迭代推理批次数据
                src_seq, tgt_seq = self.tool.batch_2_tensor(batch_dat)  # 获得输入序列和实际目标序列
                src_pos = self.tool.seq_2_pos(src_seq)  # 得到输入序列的pos位置向量
                all_pre_seq, all_pre_seq_p = infer.translate_batch(src_seq, src_pos)  # 获得所有预测的结果和对应的概率

                for index, pre_seq in enumerate(all_pre_seq):
                    src_word_seq = index_2_word(source_lang, src_seq[index])  # 清洗输入序列并转化为字符
                    tgt_word_seq = index_2_word(target_lang, tgt_seq[index])  # 清洗目标序列并转化为字符
                    for seq in pre_seq:  # 清洗预测序列并转化为字符
                        new_seq = []
                        for i in seq:
                            if i != config.sos and i != config.eos and i != config.pad:
                                new_seq.append(i)
                        if type(target_lang) != dict:
                            pre_word_seq = [target_lang.index2word[idx] for idx in new_seq]
                        else:
                            pre_word_seq = [target_lang['index2word'][idx] for idx in new_seq]

                    f.write('输入序列->：' + ' '.join(src_word_seq) + '\n')  # 写入输入序列
                    f.write('->预测序列：' + ' '.join(pre_word_seq) + '\n')  # 写入预测序列
                    f.write('==目标序列：' + ' '.join(tgt_word_seq) + '\n\n')  # 写入实际序列

        print('推理预测序列完毕！')