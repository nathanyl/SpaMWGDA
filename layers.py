import math
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, active=True):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            if active:
                output = F.relu(output)
                return output + self.bias
            else:
                return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphDeConvolution(Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        '''
        input_dim 是输入特征的维度，
        output_dim 是输出特征的维度，
        use_bias 是一个布尔值，表示是否使用偏置项。
        '''
        super(GraphDeConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        # 创建名为 weight 的可学习参数，并将其保存在类的实例中。这个权重矩阵的形状为 (input_dim, output_dim)。
        self.weight = Parameter(torch.Tensor(input_dim, output_dim))
        # 根据 use_bias 的取值来决定是否创建偏置项参数。如果 use_bias 为真，则创建名为 bias 的可学习参数，并将其保存在类的实例中；否则将 bias 注册为 None。
        if self.use_bias:
            self.bias = Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        # 调用 reset_parameters 方法，用于初始化权重和偏置参数。
        self.reset_parameters()  # 初始化w

    def reset_parameters(self):
        # 采用均匀分布的方式初始化权重。
        init.kaiming_uniform_(self.weight)
        # 将偏置项初始化为全零，即将所有偏置项的值设定为0。
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, feature_ori, feature_aug, adjacency):
        # 原始特征 feature_ori 与权重矩阵 self.weight 的矩阵乘积，得到支持特征 support_ori。
        support_ori = torch.mm(feature_ori, self.weight)
        # 利用稀疏矩阵乘法 spmm 计算了邻接矩阵 adjacency 与支持特征 support_ori 的乘积，得到原始特征经过卷积层后的输出 output_ori。
        output_ori = torch.spmm(adjacency, support_ori)
        """输入增强矩阵，输出卷积层降维结果"""
        support_aug = torch.mm(feature_aug, self.weight)
        output_aug = torch.spmm(adjacency, support_aug)

        if self.use_bias:
            output_ori += self.bias
            output_aug += self.bias
        # 对输出进行了激活函数处理，使用了 ReLU 激活函数，将输出进行了非线性变换。
        output_ori = F.relu(output_ori)
        output_aug = F.relu(output_aug)
        # 返回经过卷积层处理后的原始输出 output_ori 和增强输出 output_aug。
        return output_ori, output_aug

    # 用于返回对象的字符串表示形式。"类名 (输入维度 -> 输出维度)"
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'

