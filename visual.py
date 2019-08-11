#coding:utf-8

from config import Config
config = Config()

import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei'] # for Chinese characters
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


class Visualization:

    def train_loss(self, t_loss):
        x = [i for i in range(len(t_loss))]

        plt.xlabel(u'Batches')
        plt.ylabel(u'Loss')
        plt.title(u'The train loss')

        plt.plot(x, t_loss, label=u'loss value', linestyle=':',
                 marker='+', color='green')
        plt.legend(loc='upper right')

        plt.savefig(config.pic_train_loss)
        plt.show()
        plt.clf()

    def val_loss(self, v_loss):
        x = [i for i in range(len(v_loss))]

        plt.xlabel(u'Batches')
        plt.ylabel(u'Loss')
        plt.title(u'The validate loss')

        plt.plot(x, v_loss, label=u'loss value', linestyle=':',
                 marker='x', color='red')
        plt.legend(loc='upper right')

        plt.savefig(config.pic_val_loss)
        plt.show()
        plt.clf()

    def ppl(self, t_ppl, v_ppl):
        x_1 = [i for i in range(len(t_ppl))]
        x_2 = [i for i in range(len(v_ppl))]

        plt.xlabel(u'Epochs')
        plt.ylabel(u'Ppl')
        plt.title(u'Train & Validate PPL')

        t = plt.plot(x_1, t_ppl, label=u'ppl value', linestyle=':', marker='+', color='green')
        v = plt.plot(x_2, v_ppl, label=u'ppl value', linestyle=':', marker='x', color='red')

        plt.legend(loc='upper right')

        plt.savefig(config.pic_train_val_ppl)
        plt.show()
        plt.clf()

    def acc(self, t_acc, v_acc):
        x_1 = [i for i in range(len(t_acc))]
        x_2 = [i for i in range(len(v_acc))]

        plt.xlabel(u'Epochs')
        plt.ylabel(u'Acc')
        plt.title(u'Train & Validate ACC')

        t = plt.plot(x_1, t_acc, label=u'train acc', linestyle=':', marker='+', color='green')
        v = plt.plot(x_2, v_acc, label=u'val acc', linestyle=':', marker='x', color='red')

        plt.legend(loc='upper right')

        plt.savefig(config.pic_train_val_acc)
        plt.show()
        plt.clf()