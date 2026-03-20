from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
    CenterSpatialCropd,
)
# 上边这些以‘d’结尾的（除了Compose）是字典，key都是‘image’，作用是根据key对CT图像进行处理

ImageTransforms = Compose(
    [
        LoadImaged(keys=["image"]),     #根据路径读取图像，存储在字典的 "image" 键下
        EnsureChannelFirstd(keys=["image"]),    #确保图像的channel在最前，(C, H, W, D)
        Orientationd(keys=["image"], axcodes="RAS"),    #统一图像的解剖学方向为RAS（Right-右, Anterior-前, Superior-上）
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),    
        #重采样体素间距；pixdim=(1.5, 1.5, 3)：将所有图像的物理分辨率统一为 1.5mm × 1.5mm × 3.0mm；mode=("bilinear")：重采样时使用双线性插值
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
        ),   
        #a_min=-1000, a_max=1000：将 HU 值限定在 [-1000, 1000] 的范围内。（在CT影像中，组织的密度用Hounsfield Unit(HU) 表示，[-1000, 1000]通常涵盖了空气到骨骼的常见范围，常用于胸部或腹部CT）
        #clip=True：截断。低于 -1000 的值会被截断为 -1000，高于 1000 的会被截断为 1000
        #b_min=0.0, b_max=1.0：将 [-1000, 1000] 范围内的值线性映射到 [0.0, 1.0]
        SpatialPadd(keys=["image"], spatial_size=[224, 224, 160]),    
        #空间填充：如果经过前面步骤处理后的图像在某个维度上小于 [224, 224, 160]，就在图像边缘用 0 填充
        CenterSpatialCropd(
            roi_size=[224, 224, 160],
            keys=["image"],
        ),
        #中心裁剪：从图像的正中心裁剪出精确的 [224, 224, 160] 大小的区域
        ToTensord(keys=["image"]),    #将Numpy数组转换为Tensor对象，可以直接送入 GPU 和网络中进行训练或推理
    ]
)
