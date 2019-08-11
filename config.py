#-*-coding:utf-8-*-

# transformer的内置参数类

import torch

r""" 注释中使用的字母代表的意思
B batch_size 批大小
L max_len 序列最大长度
D dimension = d_mode 模型维度
d dimension = d_q, d_k, d_v 三个遍历query，key和value的维度 
H heads h次线性映射
"""

class Config():

    def __init__(self, d_mode=512, d_q=64, heads=8, d_ff=2048, dropout=0.1, layers=6):
        """
        Transformer模型基本参数
        :param d_mode:
        :param d_q:
        :param heads:
        :param d_ff:
        :param dropout:
        :param layers:
        """
        self.train_input = './data/train.en'
        self.train_target = './data/train.de'
        self.val_input = './data/val.en'
        self.val_target = './data/val.de'
        self.test_input = './data/test.en'
        self.test_target = './data/test.de'

        self.d_model = d_mode  # 模型维度
        self.d_q = d_q  # query变量维度
        self.d_k = d_q  # key变量维度
        self.d_v = d_q  # value变量维度
        self.heads = heads  # 多头注意力中h次进行query，key和value的线性映射
        self.d_ff = d_ff  # 按位置前馈层的内层维度
        self.layers = layers  # 编码器和解码迭代计算层数
        self.dropout = dropout  # dropout大小

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备选择
        self.use_gpu = True if torch.cuda.is_available() else False  # 设备选择
        self.multi_gpu = False  # 多GPU

        self.min_len = 0  # 序列最短长度
        self.max_len = 50  # 序列最大长度
        self.min_word_count = 5  # 输入和目标字典中词频最少限制数
        self.pad = 0  # 序列填充符及位置
        self.unk = 1  # 序列unk未知token符及位置
        self.sos = 2  # 序列开始符及位置
        self.eos = 3  # 序列终止符及位置

        # batch_size和warmup_step的对应关系，
        # 需要按照此对应表，本复现模型损失才能正常下降
        # ...32-16,000; 64-8,000; 128-4,000; 256-2,000; 512-1,000; 1024-500...
        self.batch_size = 128  # 训练批大小
        self.warmup_step = 4000  # 模型预热步
        self.epochs = 100  # 训练轮次
        self.loss_cal = 'sum'  # 损失是采用总和sum还是平均mean

        self.beam_search_left = 1  # beam search后留下top句子的数量
        self.beam_size = 5  # beam search搜索宽度
        self.infer_batch = 32  # 推理测试的批大小

        self.log = True  # 是否记录损失
        self.save_trained_model = False  # 是否保存训练模型最佳点
        self.save_trained_model_type = 'best'  # 保存模型最佳点的方式
        self.checkpoint = './results/best_model.chkpt'  # 模型最佳保存点
        self.save_data = './results/words_data.pt'  # 原始数据保存点

        self.visual = True  # 训练&验证的loss，ppl，acc三个值的可视化
        self.use_visdom = False  # 使用Visdom画图
        self.use_tensorboard = False  # TODO 使用tensorboard画图
        self.use_tensorboard_X = False  # TODO 使用tensorboardX画图

        self.pic_train_loss = './pictures/train_loss.png'
        self.pic_val_loss = './pictures/val_loss.png'
        self.pic_train_val_ppl = './pictures/train_val_ppl.png'
        self.pic_train_val_acc = './pictures/train_val_acc.png'

        self.__check()  # 参数检查

    def __check(self):
        """
        检查d_model，d_q，d_k，d_v，heads，d_ff，这几者之间关系
        :return: True/False
        """
        assert self.d_ff / self.d_model == 4, \
            '模型维度d_model和按位置前馈层的内层d_ff不匹配！'

        assert self.d_q == self.d_k or self.d_q == self.d_v or self.d_k == self.d_v, \
            '模型d_q，d_k，d_v三个变量维度维度不相等！'

        # TODO 待增加更多的参数检查断言

    # 训练中，使用命令行参数给预定的变量赋值
    def args_2_variable_train(self, args):
        self.d_model = args.d_model
        self.d_q = args.d_q
        self.d_k = args.d_k
        self.d_v = args.d_v
        self.heads = args.heads
        self.layers = args.layers
        self.d_ff = args.d_ff
        self.dropout = args.dropout

        self.epochs = args.epochs
        self.batch_size = args.train_batch_size
        self.warmup_step = args.warm_up_step
        self.beam_search_left = args.beam_search
        self.loss_cal = args.cal_loss
        self.min_word_count = args.min_word_count

    # 推理中，使用命名行参数给预定的变量赋值
    def args_2_variable_infer(self, args):
        pass
