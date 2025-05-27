from typing import Union

import pandas as pd
import numpy as np
import datetime
from einops import rearrange

from torch.utils.data import Dataset as TorchDataset
from torch import nn
from torchvision import transforms

from .sevir_dataloader import SEVIRDataLoader
from .data_utils.augmentation import TransformsFixRotation


class SEVIRTorchDataset(TorchDataset):
    orig_dataloader_layout = "NHWT"
    orig_dataloader_squeeze_layout = orig_dataloader_layout.replace("N", "")
    aug_layout = "THW"

    def __init__(
        self,
        
        # region Data Formating
        seq_len: int = 25, raw_seq_len: int = 49,stride: int = 12,
        sample_mode: str = "sequent",
        layout: str = "THWC",
        # endregion
        
        split_mode: str = "uneven", 
        
        # region SEVIR Config
        sevir_catalog: Union[str, pd.DataFrame] = None,
        sevir_data_dir: str = None,
        start_date: datetime.datetime = None,
        end_date: datetime.datetime = None,
        datetime_filter = None,
        catalog_filter = "default",
        # endregion
        
        shuffle: bool = False,
        shuffle_seed: int = 1,
        output_type = np.float32,
        
        preprocess: bool = True,
        rescale_method: str = "01",
        verbose: bool = False,
        aug_mode: str = "0",
        ret_contiguous: bool = True
    ):
        super(SEVIRTorchDataset, self).__init__()
        self.layout = layout.replace("C", "1")
        self.ret_contiguous = ret_contiguous
        self.sevir_dataloader = SEVIRDataLoader(
            seq_len=seq_len, raw_seq_len=raw_seq_len, stride=stride,
            sample_mode=sample_mode,
            batch_size=1,
            layout=self.orig_dataloader_layout,
            
            num_shard=1, rank=0, split_mode=split_mode,
            
            data_types=["vil", ],
            sevir_catalog=sevir_catalog,
            sevir_data_dir=sevir_data_dir,
            start_date=start_date,
            end_date=end_date,
            datetime_filter=datetime_filter,
            catalog_filter=catalog_filter,
            
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            output_type=output_type,
            
            preprocess=preprocess,
            rescale_method=rescale_method,
            downsample_dict=None,
            verbose=verbose)
        self._set_aug_transformation(aug_mode)
        
        
    def _set_aug_transformation(self,aug_mode:str):
        self.aug_mode = aug_mode
        if aug_mode == "0":
            self.aug = lambda x:x
        elif aug_mode == "1":
            self.aug = nn.Sequential(
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=180),
            )
        elif aug_mode == "2":
            self.aug = nn.Sequential(
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                TransformsFixRotation(angles=[0, 90, 180, 270]),
            )
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        data_dict = self.sevir_dataloader._idx_sample(index=index)
        data = data_dict["vil"].squeeze(0)
        if self.aug_mode != "0":
            data = rearrange(data, f"{' '.join(self.orig_dataloader_squeeze_layout)} -> {' '.join(self.aug_layout)}")
            data = self.aug(data)
            data = rearrange(data, f"{' '.join(self.aug_layout)} -> {' '.join(self.layout)}")
        else:
            data = rearrange(data, f"{' '.join(self.orig_dataloader_squeeze_layout)} -> {' '.join(self.layout)}")
        if self.ret_contiguous:
            return data.contiguous()
        else:
            return data

    def __len__(self):
        return self.sevir_dataloader.__len__()
    
if __name__ == "__main__":
    torch_ds = SEVIRTorchDataset()
    print(len(torch_ds))
    for idx,sample in enumerate(torch_ds):
        if idx==5:break
        print(sample.shape)