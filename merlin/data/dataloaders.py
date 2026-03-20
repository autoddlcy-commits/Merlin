import torch
import monai
from copy import deepcopy
import shutil
import tempfile
from pathlib import Path
from typing import List
from monai.utils import look_up_option
from monai.data.utils import SUPPORTED_PICKLE_MOD

from merlin.data.monai_transforms import ImageTransforms


class CTPersistentDataset(monai.data.PersistentDataset):
    #data：待处理的数据清单（N张CT的路径）
    #transform：./monai_transformer里定义的ImageTransforms
    #cache_dir：存储处理好的data（tensor张量），一个个.pt文件【Pytorch的pickle 的技术，可以通过torch.save()把PyTorch里的任何东西存成.pt文件】
    def __init__(self, data, transform, cache_dir=None):
        #继承monai.data.PersistentDataset初始化方法：
        #    将变量 data 赋值给内部的 self.data 变量
        #    将变量transformer拆成：1._pre_transform  2._post_transform
        #        1._pre_transform：确定性的、结果不会变的；比如 LoadImaged 读取、Spacingd 重采样、CenterSpatialCropd 裁剪；被缓存
        #        2._post_transform：带有随机性质的，通常以 Rand 开头；比如 RandRotated 随机旋转图片；不被缓存，在每次送入模型前实时计算
        super().__init__(data=data, transform=transform, cache_dir=cache_dir)

        print(f"Size of dataset: {self.__len__()}\n")

    def _cachecheck(self, item_transformed):
        hashfile = None
        _item_transformed = deepcopy(item_transformed)
        image_data = {
            "image": item_transformed.get("image")
        }  # Assuming the image data is under the 'image' key

        if self.cache_dir is not None and image_data is not None:
            data_item_md5 = self.hash_func(image_data).decode(
                "utf-8"
            )  # Hash based on image data
            hashfile = self.cache_dir / f"{data_item_md5}.pt"

        if hashfile is not None and hashfile.is_file():
            cached_image = torch.load(hashfile, weights_only=False)
            _item_transformed["image"] = cached_image
            return _item_transformed

        _image_transformed = self._pre_transform(image_data)["image"]
        _item_transformed["image"] = _image_transformed
        if hashfile is None:
            return _item_transformed
        try:
            # NOTE: Writing to a temporary directory and then using a nearly atomic rename operation
            #       to make the cache more robust to manual killing of parent process
            #       which may leave partially written cache files in an incomplete state
            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_hash_file = Path(tmpdirname) / hashfile.name
                torch.save(
                    obj=_image_transformed,
                    f=temp_hash_file,
                    pickle_module=look_up_option(
                        self.pickle_module, SUPPORTED_PICKLE_MOD
                    ),
                    pickle_protocol=self.pickle_protocol,
                )
                if temp_hash_file.is_file() and not hashfile.is_file():
                    # On Unix, if target exists and is a file, it will be replaced silently if the user has permission.
                    # for more details: https://docs.python.org/3/library/shutil.html#shutil.move.
                    try:
                        shutil.move(str(temp_hash_file), hashfile)
                    except FileExistsError:
                        pass
        except PermissionError:  # project-monai/monai issue #3613
            pass
        return _item_transformed

    def _transform(self, index: int):
        pre_random_item = self._cachecheck(self.data[index])
        return self._post_transform(pre_random_item)


class DataLoader(monai.data.DataLoader):
    def __init__(
        self,
        datalist: List[dict],
        cache_dir: str,
        batchsize: int,
        shuffle: bool = True,
        num_workers: int = 0,
    ):
        self.datalist = datalist
        self.cache_dir = cache_dir
        self.batchsize = batchsize
        self.dataset = CTPersistentDataset(
            data=datalist,
            transform=ImageTransforms,
            cache_dir=cache_dir,
        )
        super().__init__(
            self.dataset,
            batch_size=batchsize,
            shuffle=shuffle,
            num_workers=num_workers,
        )
