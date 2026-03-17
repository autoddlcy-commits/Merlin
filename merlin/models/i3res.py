"""Code adapted from https://github.com/hassony2/inflated_convnets_pytorch"""

import math

import torch
import torch.utils.checkpoint as checkpoint

from merlin.models import inflate


class I3ResNet(torch.nn.Module):
    def __init__(
        self,
        resnet2d, #传入的2D模型
        frame_nb=16,
        class_nb=1000,
        conv_class=False,
        return_skips=False,
        ImageEmbedding=False,
        PhenotypeCls=False,
        FiveYearPred=False,
    ):
        """
        Args:
            conv_class: Whether to use convolutional layer as classifier to
                adapt to various number of frames
        """
        super(I3ResNet, self).__init__()
        self.return_skips = return_skips
        self.conv_class = conv_class  # 是否使用卷积层作为分类器，以适应不同数量的帧，bool型。
        self.ImageEmbedding = ImageEmbedding
        self.PhenotypeCls = PhenotypeCls
        self.FiveYearPred = FiveYearPred
        #对传入的2D模型进行3D膨胀
        #ResNet结构分为1.和2.，分别进行膨胀
        #1.conv1，bn1，relu，maxpool
        self.conv1 = inflate.inflate_conv(
            resnet2d.conv1, time_dim=3, time_padding=1, center=True
        )
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(
            resnet2d.maxpool, time_dim=3, time_padding=1, time_stride=2
        )
        #2.由Bottleneck组成的layer：layer1 到 layer4【inflate_reslayer函数位于该文件，而不是inflate文件】
        self.layer1 = inflate_reslayer(resnet2d.layer1)
        self.layer2 = inflate_reslayer(resnet2d.layer2)
        self.layer3 = inflate_reslayer(resnet2d.layer3)
        self.layer4 = inflate_reslayer(resnet2d.layer4)

        if conv_class:
            self.avgpool = inflate.inflate_pool(resnet2d.avgpool, time_dim=1)
            # 【头 1：预测 EHR 的分类头】
            self.classifier = torch.nn.Conv3d(
                in_channels=2048,
                out_channels=class_nb, #外部输入是1692，表型数
                kernel_size=(1, 1, 1),
                bias=True,
            )
            # 【头 2：生成图像特征的降维头】
            self.contrastive_head = torch.nn.Conv3d(
                in_channels=2048, out_channels=512, kernel_size=(1, 1, 1), bias=True
            )    #固定输出 512 维，为了和文本对齐
        else:
            final_time_dim = int(math.ceil(frame_nb / 16))
            self.avgpool = inflate.inflate_pool(
                resnet2d.avgpool, time_dim=final_time_dim
            )
            self.fc = inflate.inflate_linear(resnet2d.fc, 1)

    def forward(self, x):
        skips = []
        x = x.permute(0, 1, 4, 2, 3)
        x = torch.cat((x, x, x), dim=1)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = self.maxpool(x)

        x = checkpoint.checkpoint(self.layer1, x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = checkpoint.checkpoint(self.layer2, x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = checkpoint.checkpoint(self.layer3, x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = checkpoint.checkpoint(self.layer4, x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))

        if self.conv_class:
            x_features = self.avgpool(x)

            if self.ImageEmbedding:
                return x_features.squeeze(2).squeeze(2).squeeze(2).unsqueeze(0)

            # 1.预测 1692 个 EHR 表型
            x_ehr = self.classifier(x_features)
            x_ehr = x_ehr.squeeze(3)
            x_ehr = x_ehr.squeeze(3)
            x_ehr = x_ehr.mean(2)
            # 2.生成 512 维的对比特征
            x_contrastive = self.contrastive_head(x_features)
            x_contrastive = x_contrastive.squeeze(3)
            x_contrastive = x_contrastive.squeeze(3)
            x_contrastive = x_contrastive.mean(2)
            if self.return_skips:
                return x_contrastive, x_ehr, skips
            else:
                if self.PhenotypeCls or self.FiveYearPred:
                    probs = torch.sigmoid(x_ehr)  #sigmoid：多标签二分类（模型对这1692个EHR特征，每一个都独立输出一个0-1之间的概率（概率> 0.5就算有））
                    return probs

                return x_contrastive, x_ehr
        else:
            x = self.avgpool(x)
            x_reshape = x.view(x.size(0), -1)
            x = self.fc(x_reshape)
        return x

#inflate_reslayer：专为resnet的layer设计的膨胀函数
#2D变为3D：class Bottleneck3d里的inflate.inflate_conv逻辑，HxW的卷积核变为TxHxW，增加“深度”维度（time_dim）
def inflate_reslayer(reslayer2d):
    reslayers3d = []
    for layer2d in reslayer2d:
        layer3d = Bottleneck3d(layer2d)
        reslayers3d.append(layer3d)
    return torch.nn.Sequential(*reslayers3d)


class Bottleneck3d(torch.nn.Module):
    #bottleneck2d：原始resnet-152的设计，主路包括3个卷积核————两头小（1x1）中间大（3x3），channel数两头大中间小；shortcut为残差连接，包括一个downsample
    #1x1卷积:混合channel信息，或改变channel数量————每次只看到单一像素，看不到邻近像素
    #    改变channel数量：节省算力，防止模型过拟合；几乎不损失图像信息————1.不是所有channel都饱含信息，有冗余和噪声 2.残差连接：out = out + residual（原始channel的信息）
    #    1x1卷积将卷积前channel赋予不同的权重然后求和，将核心特征集中在卷积后的channel里
    #3x3卷积:提取图片边缘、纹理等空间特征————可以看到中心像素的邻近像素；膨胀后有了T维度，能看到当前切片的上下切片
    def __init__(self, bottleneck2d):
        super(Bottleneck3d, self).__init__()

        spatial_stride = bottleneck2d.conv2.stride[0]

        self.conv1 = inflate.inflate_conv(bottleneck2d.conv1, time_dim=1, center=True)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)

        self.conv2 = inflate.inflate_conv(
            bottleneck2d.conv2,
            time_dim=3,
            time_padding=1,
            time_stride=spatial_stride,
            center=True,
        )
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)

        self.conv3 = inflate.inflate_conv(bottleneck2d.conv3, time_dim=1, center=True)
        self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)

        self.relu = torch.nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = inflate_downsample(
                bottleneck2d.downsample, time_stride=spatial_stride
            )
        else:
            self.downsample = None

        self.stride = bottleneck2d.stride

    def forward(self, x):
        def run_function(input_x):
            out = self.conv1(input_x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)
            return out

        residual = x    # 1.把原始输入 x 存到备用变量 residual 里

        # shortcut
        if self.downsample is not None:    # 2.判断是否进行downsample【这里的self.downsample已经转为3D的了】
            residual = self.downsample(x)  【可以写在x进入主路之前，因为HW和channel的操作在 __init__ 被执行时已经定好】

        #主路                               # 3.主路进行3次卷积
        if x.requires_grad:
            #checkpoint 优化显存：时间换空间————正向传播时，不保存这些卷积层的中间计算结果（从而省下大量显卡内存）；等到反向传播算梯度时，再临时重新算一遍
            out = checkpoint.checkpoint(run_function, x)
        else:
            out = run_function(x)

        # 4.shortcut和主路汇合
        out = out + residual
        out = self.relu(out)
        return out

#下采样的2D到3D转换函数【当 downsample2d 不是 None 时】
def inflate_downsample(downsample2d, time_stride=1):
    # downsample2d：原始resnet-152的下采样模块，包括两个；downsample2d[0]：1x1卷积层，减小长宽，放大通道数；downsample2d[1]：BN层
    downsample3d = torch.nn.Sequential(
        inflate.inflate_conv(
            downsample2d[0], time_dim=1, time_stride=time_stride, center=True
        ),
        inflate.inflate_batch_norm(downsample2d[1]),
    )
    return downsample3d
