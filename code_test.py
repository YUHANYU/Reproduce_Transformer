# loss, acc, ppl结果可视化

from config import Config
config = Config()

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # for Chinese characters
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
import numpy as np

# # loss
# train_loss = [10 + i * 0.01 for i in range(100)]
# train_loss_x = [i for i in range(100)]
#
# plt.xlabel('Step(batch num)')
# plt.ylabel('Loss value')
# plt.title('The training loss')
# plt.plot(train_loss_x, train_loss, label='train loss',
#          linestyle=':', marker='+', color='green')
# plt.legend(loc='upper right')
#
# plt.show()


class Visualization:

    def train_loss(self, t_loss):
        x = [i for i in range(len(t_loss))]

        plt.xlabel('累积批次')
        plt.ylabel('损失值')
        plt.title('训练损失图')

        plt.plot(x, t_loss, label='训练损失值', linestyle=':',
                 marker='+', color='green')
        plt.legend(loc='upper right')

        plt.savefig(config.pic_train_loss)
        plt.show()

    def val_loss(self, v_loss):
        x = [i for i in range(len(v_loss))]

        plt.xlabel('累积批次')
        plt.ylabel('损失值')
        plt.title('验证损失图')

        plt.plot(x, v_loss, label='验证损失值', linestyle=':',
                 marker='x', color='red')
        plt.legend(loc='upper right')

        plt.savefig(config.pic_val_loss)
        plt.show()

    def ppl(self, t_ppl, v_ppl):
        x_1 = [i for i in range(len(t_ppl))]
        x_2 = [i for i in range(len(v_ppl))]

        plt.xlabel('累积批次')
        plt.ylabel('PPL值')
        plt.title('训练&验证PPL图')

        t = plt.plot(x_1, t_ppl, label='训练PPL', linestyle=':', marker='+', color='green')
        v = plt.plot(x_2, v_ppl, label='验证PPL', linestyle=':', marker='x', color='red')

        plt.legend(loc='upper right')

        plt.savefig(config.pic_train_val_ppl)
        plt.show()

    def acc(self, t_acc, v_acc):
        x_1 = [i for i in range(len(t_acc))]
        x_2 = [i for i in range(len(v_acc))]

        plt.xlabel('累积批次')
        plt.ylabel('acc值')
        plt.title('训练&验证acc图')

        t = plt.plot(x_1, t_acc, label='训练ACC', linestyle=':', marker='+', color='green')
        v = plt.plot(x_2, v_acc, label='验证ACC', linestyle=':', marker='x', color='red')

        plt.legend(loc='upper right')

        plt.savefig(config.pic_train_val_acc)
        plt.show()
