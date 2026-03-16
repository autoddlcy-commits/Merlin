""" Code adapted from https://github.com/hassony2/inflated_convnets_pytorch """

import torch
from torch.nn import Parameter

#inflate_conv: 2D 卷积膨胀成3D 卷积，增加深度/时间维度 T ，也就是 time_dim；分为1.中心法和2.复制法
def inflate_conv(
    conv2d, time_dim=3, time_padding=0, time_stride=1, time_dilation=1, center=False
):
    # To preserve activations, padding should be by continuity and not zero
    # or no padding in time dimension
    if conv2d.kernel_size[0] == 7:
        kernel_dim = (3, 7, 7)
        padding = (1, 3, 3)
        stride = (1, 2, 2)
        dilation = (1, 1, 1)
        conv3d = torch.nn.Conv3d(
            conv2d.in_channels,
            conv2d.out_channels,
            kernel_dim,
            padding=padding,
            dilation=dilation,
            stride=stride,
        )
        # Repeat filter time_dim times along time dimension
        weight_2d = conv2d.weight.data
        #1.中心填入法 (center=True 1)
        if center:
            # 建一个全为 0 的立体方块
            weight_3d = torch.zeros(*weight_2d.shape) #全0的2D
            weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1) #复制 time_dim 份，叠成 time_dim 层
            # 找到中间一层
            middle_idx = time_dim // 2
            # 中间层填入原本 2D 的权重
            weight_3d[:, :, middle_idx, :, :] = weight_2d
        #原理：当这个 3D 卷积核去扫 CT 图像时，因为它上下两层全是 0，所以它实际上只看到了当前这一层的图像，计算结果和原来的 2D 模型一模一样。
        #     随着后续训练，那些 0 的地方才会慢慢更新出数值，学会看上下层的信息。
        #2.层层复制法 (center=False 1)
        else:
            # 把原来的 2D 权重，直接复制 time_dim 份，叠成 time_dim 层
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            # 把所有权重的值除以 time_dim：上一步后参数量级变成 time_dim 倍，需要除以 time_dim ，防止模型爆炸
            weight_3d = weight_3d / time_dim

        # Assign new params
        conv3d.weight = Parameter(weight_3d)
        conv3d.bias = conv2d.bias
    else:
        kernel_dim = (time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1])
        padding = (time_padding, conv2d.padding[0], conv2d.padding[1])
        stride = (time_stride, conv2d.stride[0], conv2d.stride[0])
        dilation = (time_dilation, conv2d.dilation[0], conv2d.dilation[1])
        conv3d = torch.nn.Conv3d(
            conv2d.in_channels,
            conv2d.out_channels,
            kernel_dim,
            padding=padding,
            dilation=dilation,
            stride=stride,
        )
        # Repeat filter time_dim times along time dimension
        weight_2d = conv2d.weight.data
        #1.中心填入法 (center=True 2)
        if center:
            weight_3d = torch.zeros(*weight_2d.shape)
            weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            middle_idx = time_dim // 2
            weight_3d[:, :, middle_idx, :, :] = weight_2d
        #2.层层复制法 (center=False 2)
        else:
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            weight_3d = weight_3d / time_dim

        # Assign new params
        conv3d.weight = Parameter(weight_3d)
        conv3d.bias = conv2d.bias
    return conv3d


def inflate_linear(linear2d, time_dim):
    """
    Args:
        time_dim: final time dimension of the features
    """
    linear3d = torch.nn.Linear(linear2d.in_features * time_dim, linear2d.out_features)
    weight3d = linear2d.weight.data.repeat(1, time_dim)
    weight3d = weight3d / time_dim

    linear3d.weight = Parameter(weight3d)
    linear3d.bias = linear2d.bias
    return linear3d

#把传入的batch2d的检查函数（4维）换成batch3d的检查函数（5维）
def inflate_batch_norm(batch2d):
    # In pytorch 0.2.0 the 2d and 3d versions of batch norm
    # work identically except for the check that verifies the
    # input dimensions 
    #2D：4 [Batch, Channel, Height, Width]
    #3D：5 [Batch, Channel, Depth, Height, Width]
    #batch_norm：按 Channel 计算所有像素求均值和方差；把数据重新拉回到均值为 0、方差为 1 的标准范围内，让网络训练更稳定。

    # BatchNorm2d 和 BatchNorm3d 的_check_input_dim函数分别只接受 4维 和 5维 的输入
    batch3d = torch.nn.BatchNorm3d(batch2d.num_features)
    # retrieve 3d _check_input_dim function
    batch2d._check_input_dim = batch3d._check_input_dim
    return batch2d


def inflate_pool(pool2d, time_dim=1, time_padding=0, time_stride=None, time_dilation=1):
    if isinstance(pool2d, torch.nn.AdaptiveAvgPool2d):
        pool3d = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
    else:
        kernel_dim = (time_dim, pool2d.kernel_size, pool2d.kernel_size)
        padding = (time_padding, pool2d.padding, pool2d.padding)
        if time_stride is None:
            time_stride = time_dim
        stride = (time_stride, pool2d.stride, pool2d.stride)
        if isinstance(pool2d, torch.nn.MaxPool2d):
            dilation = (time_dilation, pool2d.dilation, pool2d.dilation)
            pool3d = torch.nn.MaxPool3d(
                kernel_dim,
                padding=padding,
                dilation=dilation,
                stride=stride,
                ceil_mode=pool2d.ceil_mode,
            )
        elif isinstance(pool2d, torch.nn.AvgPool2d):
            pool3d = torch.nn.AvgPool3d(kernel_dim, stride=stride)
        else:
            raise ValueError(
                "{} is not among known pooling classes".format(type(pool2d))
            )

    return pool3d
