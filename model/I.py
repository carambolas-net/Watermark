#todo 最简单的神经网络，输入两个张量，输出两个张量，保持结果不变化
import torch

class I(torch.nn.Module):
    def __init__(self, config):
        super(I, self).__init__()
        # 占位参数，使优化器有参数可优化
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        # dummy参与计算以保持梯度流动
        return x1 + self.dummy * 0, x2 + self.dummy * 0