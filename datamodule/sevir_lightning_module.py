from typing import Tuple,Literal,Optional
import os
import datetime

import numpy as np
import torch
from torch.utils.data import random_split,DataLoader

from lightning import LightningDataModule,seed_everything

from .sevir_const import (
    SEVIR_ROOT_DIR,SEVIR_ROOT_LR_DIR,
    SEVIR_CATALOG,SEVIR_LR_CATALOG,
    SEVIR_DATA_DIR,SEVIR_LR_DATA_DIR,
    SEVIR_RAW_SEQ_LEN,SEVIR_LR_RAW_SEQ_LEN,
    SEVIR_INTERVAL_REAL_TIME,SEVIR_LR_INTERVAL_REAL_TIME,
    SEVIR_H_W_SIZE,SEVIR_LR_H_W_SIZE
)
from .sevir_torch_dataset import SEVIRTorchDataset

class SEVIRLightningDataModule(LightningDataModule):

    def __init__(
        self,
        seq_len: int = 25,stride: int = 12,
        sample_mode: str = "sequent",
        layout: str = "NTHWC",
        
        output_type = np.float32,
        preprocess: bool = True,
        rescale_method: str = "01",
        verbose: bool = False,
        aug_mode: str = "0",
        ret_contiguous: bool = True,
        
        # region Datamodule Config
        dataset_name:Literal['sevir','sevirlr'] = "sevir",
        sevir_dir: str = None,
        start_date: Tuple[int] = None,
        train_test_split_date: Tuple[int] = (2019, 6, 1),
        end_date: Tuple[int] = None,
        val_ratio: float = 0.1,
        batch_size: int = 1,
        num_workers: int = 1,
        seed: int = 0,
        # endregion
    ):
        super(SEVIRLightningDataModule, self).__init__()
        
        # region Data Formatting
        self.seq_len = seq_len
        self.stride = stride
        self.sample_mode = sample_mode
        
        assert layout[0] == "N"
        self.layout = layout.replace("N", "")
        # endregion
        
        # region Operation Config
        self.output_type = output_type
        self.preprocess = preprocess
        self.rescale_method = rescale_method
        self.verbose = verbose
        self.aug_mode = aug_mode
        self.ret_contiguous = ret_contiguous
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        # endregion
        
        self._config_sevir(sevir_dir,dataset_name)
        
        # train val test split
        self.start_date = datetime.datetime(*start_date) \
            if start_date is not None else None
        self.train_test_split_date = datetime.datetime(*train_test_split_date) \
            if train_test_split_date is not None else None
        self.end_date = datetime.datetime(*end_date) \
            if end_date is not None else None
        self.val_ratio = val_ratio
        
    def _config_sevir(self,sevir_dir:Optional[str], dataset_name:Literal['sevir','sevirlr']):
        assert dataset_name in ['sevir','sevirlr'], f"Unknown dataset configuration {dataset_name}"
        sevir_dir = os.path.abspath(sevir_dir) if sevir_dir is not None else None
        if dataset_name == "sevir":
            if sevir_dir is None:
                sevir_dir = SEVIR_ROOT_DIR
            catalog_path = SEVIR_CATALOG
            raw_data_dir = SEVIR_DATA_DIR
            raw_seq_len = SEVIR_RAW_SEQ_LEN
            interval_real_time = SEVIR_INTERVAL_REAL_TIME
            img_height,img_width = SEVIR_H_W_SIZE
        elif dataset_name == "sevirlr":
            if sevir_dir is None:
                sevir_dir = SEVIR_ROOT_LR_DIR
            catalog_path = SEVIR_LR_CATALOG
            raw_data_dir = SEVIR_LR_DATA_DIR
            raw_seq_len = SEVIR_LR_RAW_SEQ_LEN
            interval_real_time = SEVIR_LR_INTERVAL_REAL_TIME
            img_height,img_width = SEVIR_LR_H_W_SIZE
        else:
            raise ValueError(f"Wrong dataset name {dataset_name}. Must be 'sevir' or 'sevirlr'.")
        self.dataset_name = dataset_name
        self.sevir_dir = sevir_dir
        self.catalog_path = catalog_path
        self.raw_data_dir = raw_data_dir
        self.raw_seq_len = raw_seq_len
        self.interval_real_time = interval_real_time
        self.img_height = img_height
        self.img_width = img_width

    def prepare_data(self) -> None:
        if os.path.exists(self.sevir_dir):
            # Further check
            assert os.path.exists(self.catalog_path), f"CATALOG.csv not found! Should be located at {self.catalog_path}"
            assert os.path.exists(self.raw_data_dir), f"SEVIR data not found! Should be located at {self.raw_data_dir}"
        else:
            raise NotImplementedError(f'Data not available in specified directory {self.sevir_dir}')

    def setup(self, stage = None) -> None:
        seed_everything(seed=self.seed)
        if stage in (None, "fit"):
            sevir_train_val = SEVIRTorchDataset(
                sevir_catalog=self.catalog_path,
                sevir_data_dir=self.raw_data_dir,
                raw_seq_len=self.raw_seq_len,
                split_mode="uneven",
                shuffle=True,
                seq_len=self.seq_len,
                stride=self.stride,
                sample_mode=self.sample_mode,
                layout=self.layout,
                start_date=self.start_date,
                end_date=self.train_test_split_date,
                output_type=self.output_type,
                preprocess=self.preprocess,
                rescale_method=self.rescale_method,
                verbose=self.verbose,
                aug_mode=self.aug_mode,
                ret_contiguous=self.ret_contiguous,
            )
            self.sevir_train, self.sevir_val = random_split(
                dataset=sevir_train_val,
                lengths=[1 - self.val_ratio, self.val_ratio],
                generator=torch.Generator().manual_seed(self.seed)
            )
        if stage in (None, "test"):
            self.sevir_test = SEVIRTorchDataset(
                sevir_catalog=self.catalog_path,
                sevir_data_dir=self.raw_data_dir,
                raw_seq_len=self.raw_seq_len,
                split_mode="uneven",
                shuffle=False,
                seq_len=self.seq_len,
                stride=self.stride,
                sample_mode=self.sample_mode,
                layout=self.layout,
                start_date=self.train_test_split_date,
                end_date=self.end_date,
                output_type=self.output_type,
                preprocess=self.preprocess,
                rescale_method=self.rescale_method,
                verbose=self.verbose,
                aug_mode="0",
                ret_contiguous=self.ret_contiguous,
            )

    def train_dataloader(self):
        return DataLoader(
            self.sevir_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.sevir_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.sevir_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    @property
    def num_train_samples(self):
        return len(self.sevir_train)

    @property
    def num_val_samples(self):
        return len(self.sevir_val)

    @property
    def num_test_samples(self):
        return len(self.sevir_test)
