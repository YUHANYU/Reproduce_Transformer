通过阅读论文***Attention is all you need***，复现**Transformer**模型。

# 已完成
- [x] 输入数据处理部分
- [x] transformer模型的训练部分
- [x] transformer模型的验证部分
- [x] transformer模型的推理部分
- [x] 增加Visdom可视化工具
- [x] 增加模型日志功能
- [x] 将当前代码拆分为各个模块
- [x] 优化模型代码，添加更多注释
- [x] 添加命令行参数模式
- [x] 添加参考论文和代码的链接
- [x] 添加对模型训练部分，测试部分困惑度PPL和准确率ACC的图
- [x] 构造输入参数约束函数

# 待完成
- [ ] 增加模型多GPU运行代码
- [ ] 数据修改word形式为sub-word形式
- [ ] 增加tensorboard可视化（pytorch 1.1.0）
- [ ] 增加tensorboardX可视化

# 运行环境
+ pytorch 1.1.0（可使用tensorboard）
+ python 3.7.0
+ visdom 0.1.8.8
+ GTX 1080Ti & GTX TITAN X

# 使用方法
  ## 直接运行方式
- 在`config.py`文件中修改模型的各个参数；
- 运行`all.py`文件

  ## 命令行方式
- 运行模型的训练&验证程序`train_val.py`，需要在`train_val.py`文件中修改各类参数；
- 模型训练完成后，会自动保存最佳模型检查点，之后运行`infer.py`，修改其中的各类参数，既可以得到最后的输入结果。

  ## 注意：以上两种方式程序运行时间较久，需保持机器正常运行和耐心等待！

# 参考
+ 该复现的Tranformer模型主要是参考论文 [*Attention is all you need*](https://arxiv.org/abs/1706.03762)
+ 代码主要参考了该博主完整的transformer代码 [*jadore801120*](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
+ 代码参考了该博主的transformer结构代码 [*luozhouyang*](https://luozhouyang.github.io/transformer/)

# 可视化

  ## base model（d_model=512， d_ff=2048，h=8, P_dropout=0.1, batch_size=128, epoch time≈00:01:40）
  ![](https://github.com/YUHANYU/Reproduce_Transformer/blob/master/base%20%26%20big%20model/base%20model/train_loss.png)![](https://github.com/YUHANYU/Reproduce_Transformer/blob/master/base%20%26%20big%20model/base%20model/val_loss.png)
  ![](https://github.com/YUHANYU/Reproduce_Transformer/blob/master/base%20%26%20big%20model/base%20model/train_val_acc.png)![](https://github.com/YUHANYU/Reproduce_Transformer/blob/master/base%20%26%20big%20model/base%20model/train_val_ppl.png)
<figure class="half">
    <img src="https://github.com/YUHANYU/Reproduce_Transformer/blob/master/base%20%26%20big%20model/base%20model/train_loss.png"><img src="https://github.com/YUHANYU/Reproduce_Transformer/blob/master/base%20%26%20big%20model/base%20model/val_loss.png">
</figure>

  ## big model（d_model=1024， d_ff=4096，h=16，P_dropout=0.3, batch_size=32, epoch time≈00:03:24）
  ![](https://github.com/YUHANYU/Reproduce_Transformer/blob/master/base%20%26%20big%20model/big%20model/train_loss.png)![](https://github.com/YUHANYU/Reproduce_Transformer/blob/master/base%20%26%20big%20model/big%20model/val_loss.png)
  ![](https://github.com/YUHANYU/Reproduce_Transformer/blob/master/base%20%26%20big%20model/big%20model/train_val_acc.png)![](https://github.com/YUHANYU/Reproduce_Transformer/blob/master/base%20%26%20big%20model/big%20model/train_val_ppl.png)
